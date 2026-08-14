"""Parallel + cached version of rdd_hyperparam_tuning.py.

Layers of acceleration (no edits to helpers.py / fct_def):

1.  Process-level parallelism: one worker per device-slot, scheduled via
    multiprocessing spawn pool (required for CUDA). PROCS_PER_GPU controls
    over-subscription per GPU.
2.  Per-worker cache of the reaction-diffusion operator `g` and the
    clipping mask, keyed by (sigma, sample_idx). These are determined
    entirely by (sigma, i) -- they do NOT depend on zeta/lambda/lambda_rd
    -- so for pilatus they're built once and reused across the
    1*7*8 = 56 configs that share each (sigma, i). This avoids the bulk
    of the CPU-bound prep cost (sparse Laplacian construction).
3.  Configs are dispatched in sigma-major order so a worker tends to
    stay in one sigma group, maximising cache hits.
4.  Optional `torch.compile` of the U-Net (TORCH_COMPILE=1) -- the
    diffusion sampler hits the model num_inference_steps * num_eval_samples
    times per config, so compiling pays back even with warmup overhead.
5.  Dropped redundant `copy.deepcopy` on GPU tensors. The noisy `y` is
    still recomputed per-iteration via `randn_like` to keep the global
    torch-RNG advance bit-equivalent to the serial reference run before
    `sample_posterior_diffpir` is called.

Env knobs
---------
PROCS_PER_GPU    workers per GPU (default 2; bump to 3-4 if VRAM allows)
CPU_WORKERS      workers when no GPU is present (default 1)
GPU_IDS          comma-separated GPU ids to use (default: all visible)
TORCH_COMPILE    if "1", wrap the model with torch.compile (default off)
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.multiprocessing as mp
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm.auto import tqdm

import reaction_diffusion.routines.tomo_fusion.functionals_definition as fct_def
import reaction_diffusion.routines.tomo_fusion.tools.helpers as tomo_helps
from helpers import *  # noqa: F401,F403  (build_schedule, MyTinyUNet, sample_posterior_diffpir, ...)


# ---- fixed configuration matching rdd_hyperparam_tuning.py ----
DIAGNOSTIC = "pilatus"
H, W = 120, 40
WEIGHTS_PATH = Path("fusion_epochs/fusion_epoch_2193.pth")
NUM_TRAIN_STEPS = 1000
NUM_EVAL_SAMPLES = 1
NUM_POSTERIOR_SAMPLES = 100
NUM_INFERENCE_STEPS = 250
SAMPLES_IDXS = np.arange(1, 100)
SEED = 42

TEST_SAMPLES_PATH = Path(
    "/home/fusiontomo/Repos/diffusion_tomography/test_set_sxr_samples/sxr_samples_with_background_coarse.npy"
)
TRIM_VALUES_PATH = Path(
    "/home/fusiontomo/Repos/diffusion_tomography/test_set_sxr_samples/trimming_values.npy"
)
PSIS_PATH = Path(
    "/home/fusiontomo/Repos/diffusion_tomography/test_set_sxr_samples/psis_coarse.npy"
)

CSV_DIR = Path("metrics_csv")


# ---- per-worker globals (populated in initializer) ----
_W = {}


def _init_worker(device_queue):
    """Pull a device assignment from the queue and load all per-worker state."""
    device_str = device_queue.get()

    # Be a polite citizen when multiple workers share a host.
    torch.set_num_threads(1)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    schedule = build_schedule(num_train_steps=NUM_TRAIN_STEPS, device=device)  # noqa: F405
    model = MyTinyUNet(in_c=1, out_c=1, n_steps=NUM_TRAIN_STEPS).to(device)  # noqa: F405

    raw_state = torch.load(WEIGHTS_PATH, map_location=device)
    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        raw_state = raw_state["state_dict"]
    if isinstance(raw_state, dict) and any(k.startswith("network.") for k in raw_state.keys()):
        net_state = {k[len("network."):]: v for k, v in raw_state.items() if k.startswith("network.")}
    else:
        net_state = raw_state
    missing, unexpected = model.load_state_dict(net_state, strict=False)
    if missing or unexpected:
        raise RuntimeError("Checkpoint/model mismatch. Check architecture or key prefix handling.")

    if os.environ.get("TORCH_COMPILE", "0") == "1":
        try:
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)
            print(f"[pid {os.getpid()}] torch.compile enabled", flush=True)
        except Exception as e:  # pragma: no cover -- best-effort
            print(f"[pid {os.getpid()}] torch.compile failed, falling back: {e}", flush=True)

    # forward model
    if DIAGNOSTIC == "sxr":
        A_path = Path("forward_model/forward_model_sxr_full_geometry.npy")
    elif DIAGNOSTIC == "dmpx":
        A_path = Path("forward_model/dmpx_geometry_matrix.npy")
    elif DIAGNOSTIC == "pilatus":
        A_path = Path("forward_model/pilatus_geometry_matrix.npy")
    else:
        raise ValueError(f"unknown diagnostic {DIAGNOSTIC}")
    A_sparse = np.load(A_path).astype(np.float32)
    A_sparse /= A_sparse.max()
    A_tomo = torch.from_numpy(A_sparse).to(device)
    A_sparse_csr = sp.csr_matrix(A_sparse)

    test_samples = (
        torch.from_numpy(np.load(TEST_SAMPLES_PATH)[SAMPLES_IDXS]).to(device).to(torch.float32)
    )
    trim_vals = (
        torch.from_numpy(np.load(TRIM_VALUES_PATH)[SAMPLES_IDXS]).to(device).to(torch.float32)
    )
    psis = np.load(PSIS_PATH)[SAMPLES_IDXS]
    y_tomo_clean = test_samples.reshape(-1, H * W) @ A_tomo.T

    _W.update(
        device=device,
        model=model,
        schedule=schedule,
        A_tomo=A_tomo,
        A_sparse_csr=A_sparse_csr,
        test_samples=test_samples,
        trim_vals=trim_vals,
        psis=psis,
        y_tomo_clean=y_tomo_clean,
        # Per-worker (sigma, i) -> {"mask_core", "clipping_mask", "g"}
        cache={},
    )
    print(f"[pid {os.getpid()}] worker initialized on {device}", flush=True)


def _get_sample_state(sigma_y_tomo, i, clip, anis_param):
    """Return cached `g`, `mask_core`, `clipping_mask` for (sigma, i).

    `g`/`mask` only depend on (sigma, i) (and in the original loop, also on
    `clip`/`anis_param`, but those are constant across all configs that
    share `sigma`). Build on miss; otherwise reuse.
    """
    cache = _W["cache"]
    key = (float(sigma_y_tomo), int(i))
    cached = cache.get(key)
    if cached is not None:
        return cached

    device = _W["device"]
    psis = _W["psis"]
    trim_vals = _W["trim_vals"]
    A_sparse_csr = _W["A_sparse_csr"]
    y_tomo_clean = _W["y_tomo_clean"]

    if clip:
        mask_core = tomo_helps.define_core_mask(
            psi=psis[i],
            dim_shape=(H, W),
            trim_values_x=trim_vals[i],
        )
        clipping_mask = torch.tensor(mask_core, dtype=torch.float32, device=device)
    else:
        mask_core = None
        clipping_mask = None

    # Reproduce the same noisy y the original loop would build at this point
    # (deterministic from seed=i + sigma + y_clean[i]). Used only as input to
    # `g`'s construction; the per-iteration loop still re-draws `y_used` via
    # randn_like to keep the torch-RNG advance bit-equivalent for the
    # subsequent diffusion sampler.
    np.random.seed(i)
    torch.manual_seed(i)
    y_for_g = (y_tomo_clean[i] + sigma_y_tomo * torch.randn_like(y_tomo_clean[i])).detach()

    _, g = fct_def.define_loglikelihoodfromdata_and_logprior(
        y_for_g.cpu().numpy().reshape(1, -1),
        psis[i],
        fwd_matrix=A_sparse_csr,
        reconstruction_shape=(1, H, W),
        sigma_err=sigma_y_tomo,
        reg_fct_type="anisotropic",
        alpha=anis_param,
        sampling=0.0125,
    )

    cached = {"mask_core": mask_core, "clipping_mask": clipping_mask, "g": g}
    cache[key] = cached
    return cached


def _process_config(cfg):
    sigma_y_tomo, clip, zeta, lambda_, lambda_rd_, anis_params_list = cfg

    clip_suffix = f"_clip{clip}" if DIAGNOSTIC == "pilatus" else ""
    csv_name = (
        f"metrics_diffusion_{DIAGNOSTIC}_{sigma_y_tomo}_zeta{zeta}"
        f"_lambda{lambda_}_lambda_rd{lambda_rd_}{clip_suffix}.csv"
    )
    csv_path = CSV_DIR / csv_name
    if csv_path.exists():
        return f"skip {csv_path.name}"

    device = _W["device"]
    model = _W["model"]
    schedule = _W["schedule"]
    A_tomo = _W["A_tomo"]
    test_samples = _W["test_samples"]
    y_tomo_clean = _W["y_tomo_clean"]

    rows = []
    for i in range(NUM_EVAL_SAMPLES):
        # Pull cached (g, mask) for (sigma, i). Cache miss only happens once
        # per (sigma, i) per worker.
        cached = _get_sample_state(sigma_y_tomo, i, clip, anis_params_list[i])
        clipping_mask = cached["clipping_mask"]
        g = cached["g"]

        # Re-seed and re-draw y -- same torch-RNG state as the serial loop
        # right before sample_posterior_diffpir consumes from it.
        np.random.seed(i)
        torch.manual_seed(i)
        y_used = y_tomo_clean[i] + sigma_y_tomo * torch.randn_like(y_tomo_clean[i])
        x_gt_i = test_samples[i]

        posterior_tomo, _, _ = sample_posterior_diffpir(  # noqa: F405
            model=model,
            schedule=schedule,
            A=A_tomo,
            y=y_used,
            image_shape=(1, H, W),
            num_samples=NUM_POSTERIOR_SAMPLES,
            num_inference_steps=NUM_INFERENCE_STEPS,
            zeta=zeta,
            sigma_y=sigma_y_tomo,
            lambda_=lambda_,
            clipping_mask=clipping_mask,
            reaction_diffusion_op=g,
            lambda_rd=lambda_rd_,
        )

        post_mean = posterior_tomo.mean(dim=0).squeeze()
        post_std = posterior_tomo.std(dim=0).squeeze()

        gt_np = x_gt_i.detach().cpu().numpy()
        mean_np = post_mean.detach().cpu().numpy()
        data_range = float(gt_np.max() - gt_np.min())

        rows.append(
            {
                "sample_idx": int(SAMPLES_IDXS[i]),
                "psnr": psnr(gt_np, mean_np, data_range=data_range),
                "ssim": ssim(gt_np, mean_np, data_range=data_range),
                "mse": float(((post_mean - x_gt_i) ** 2).mean().item()),
                "post_std_mean": float(post_std.mean().item()),
                "sigma_used": float(sigma_y_tomo),
                "zeta": zeta,
                "lambda_": lambda_,
                "lambda_rd": lambda_rd_,
                "anis_param": anis_params_list[i],
                "clip": clip,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return f"saved {csv_path.name}"


def _build_configs():
    if DIAGNOSTIC == "sxr":
        lambda_values = [100, 200, 300, 400, 500]
        zeta_values = [0.2, 0.4, 0.6, 0.8, 1.0]
        clip_values = [False]
        noise_levels = [0.05, 0.25, 0.5]
        lambda_rd_values = [0.0]
    elif DIAGNOSTIC == "dmpx":
        lambda_values = [50, 100, 200, 300, 400, 500]
        zeta_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
        clip_values = [False]
        noise_levels = [0.1, 0.5, 1.0]
        lambda_rd_values = [0.0]
    elif DIAGNOSTIC == "pilatus":
        lambda_values = [1, 5, 10, 25, 50, 100, 200]
        zeta_values = [0.9]
        clip_values = [True]
        noise_levels = [0.07, 0.35, 0.7]
        lambda_rd_values = list(np.logspace(-9, -2, 8))
    else:
        raise ValueError(f"unknown diagnostic {DIAGNOSTIC}")

    anis_by_noise = {}
    for sigma in noise_levels:
        anis_csv = f"metrics_csv_phantom_analysis_rd/metrics_rd_pilatus_{sigma}_false.csv"
        if os.path.exists(anis_csv):
            df_anis = pd.read_csv(anis_csv)
            anis_by_noise[sigma] = df_anis["best_anis_param"][:NUM_EVAL_SAMPLES].tolist()
        else:
            print(f"File {anis_csv} not found.")
            anis_by_noise[sigma] = [None] * NUM_EVAL_SAMPLES

    # Sigma-major ordering keeps each worker inside one (sigma, i) cache
    # window for as long as possible.
    configs = []
    for sigma in noise_levels:
        anis_list = anis_by_noise[sigma]
        for clip in clip_values:
            for zeta in zeta_values:
                for lam in lambda_values:
                    for lam_rd in lambda_rd_values:
                        configs.append((sigma, clip, zeta, lam, lam_rd, anis_list))
    return configs


def _resolve_devices():
    if not torch.cuda.is_available():
        n_cpu = int(os.environ.get("CPU_WORKERS", "1"))
        return ["cpu"] * max(1, n_cpu)

    if "GPU_IDS" in os.environ and os.environ["GPU_IDS"].strip():
        gpu_ids = [int(x) for x in os.environ["GPU_IDS"].split(",") if x.strip() != ""]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))

    procs_per_gpu = int(os.environ.get("PROCS_PER_GPU", "2"))
    return [f"cuda:{g}" for g in gpu_ids for _ in range(procs_per_gpu)]


def main():
    print("device probe:", "cuda" if torch.cuda.is_available() else "cpu")
    CSV_DIR.mkdir(exist_ok=True)
    assert WEIGHTS_PATH.exists(), f"Missing checkpoint: {WEIGHTS_PATH}"

    configs = _build_configs()
    print(f"Total configs: {len(configs)}")

    devices = _resolve_devices()
    n_workers = len(devices)
    print(f"Using {n_workers} worker(s): {devices}")

    ctx = mp.get_context("spawn")
    device_queue = ctx.Queue()
    for d in devices:
        device_queue.put(d)

    # chunksize=1 keeps load-balancing tight; with sigma-major dispatch and
    # FIFO queueing each worker still drains contiguous (sigma, i) groups in
    # practice, so the cache stays warm.
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(device_queue,),
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_config, configs, chunksize=1),
            total=len(configs),
            desc="configs",
        ):
            tqdm.write(result)


if __name__ == "__main__":
    main()
