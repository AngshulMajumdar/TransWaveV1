
# =============================================================================
# TRANSWAVE-MEYER-96M-4.5B — PRETRAINING ONLY
#
# One raw byte = one language-model token.
# No BPE / SentencePiece / offline tokenization.
# The learned 8-byte lifting tokenizer is inside CTWModel itself.
#
# This cell ALWAYS trains to the full 4.5B exposure. There is no early stopping.
# It is restart-safe and writes recovery checkpoints to Google Drive.
# =============================================================================

from __future__ import annotations

import os
import gc
import json
import math
import time
import random
import shutil
import hashlib
from pathlib import Path

import numpy as np
import torch


# -----------------------------------------------------------------------------
# EXACT RELEASE TARGET
# -----------------------------------------------------------------------------

RELEASE_NAME = "TransWave-Meyer-96M-4.5B"
EXPECTED_ARCH = "causal_transwave_blocklifting_meyer_byte_lm_g4_v2"
EXPECTED_PARAMS = 96_130_040

# 137,329 complete 32,768-byte contexts.
TRAIN_TOKENS = 4_499_996_672
assert TRAIN_TOKENS == 137_329 * 32_768

BASE_LR = 1.0e-4
MIN_LR = 1.0e-5
WARMUP_FRACTION = 0.01
WEIGHT_DECAY = 0.0
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1.0e-8
GRAD_CLIP = 1.0

SAVE_EVERY_STEPS = 25
VAL_WINDOWS_PER_DATASET = 8

MODEL_SEED = 20261010
ORDER_SEED = 20261011

VRAM_HEADROOM_GB = 3.0
BATCH_SEARCH_START = 176
BATCH_SEARCH_STEP = 8
BATCH_SEARCH_CEILING = 224

DATASETS = (
    "arxiv",
    "fineweb_edu",
    "pg19",
    "wikitext103",
)

if ARCH != EXPECTED_ARCH:
    raise RuntimeError(
        f"Wrong architecture active: {ARCH!r}; expected {EXPECTED_ARCH!r}."
    )

assert RAW_BYTES_PER_SEQUENCE == 32_768
assert BYTE_VOCAB == 256
assert MODEL_DIM == 768
assert N_LAYERS == 12


# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

DRIVE = Path("/content/drive/MyDrive")
RAW_ROOT = DRIVE / "spliNet" / "raw_text"

RUN_ROOT = (
    DRIVE
    / "TransWave_Meyer_Context"
    / "hf_pretraining"
    / RELEASE_NAME
)

LOCAL_ROOT = Path("/content/transwave_4p5b_raw")

RUN_ROOT.mkdir(parents=True, exist_ok=True)
LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

LATEST = RUN_ROOT / "training_latest.pt"
FINAL = RUN_ROOT / "training_final.pt"
RESULT = RUN_ROOT / "result.json"
PROTOCOL = RUN_ROOT / "training_config.json"
HISTORY_FILE = RUN_ROOT / "training_history.json"
DATA_MANIFEST = RUN_ROOT / "data_manifest.json"
BATCH_PLAN_FILE = RUN_ROOT / "g4_batch_plan.json"
HF_EXPORT = RUN_ROOT / "hf_export"

HF_EXPORT.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def _atomic_json(obj, path: Path):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _atomic_torch_save(obj, path: Path):
    # Serialize to local SSD first, then copy atomically into Drive.
    local_tmp = Path("/content") / (path.name + ".localtmp")
    drive_tmp = Path(str(path) + ".tmp")
    local_tmp.unlink(missing_ok=True)
    drive_tmp.unlink(missing_ok=True)

    torch.save(obj, local_tmp)
    shutil.copyfile(local_tmp, drive_tmp)
    os.replace(drive_tmp, path)
    local_tmp.unlink(missing_ok=True)


def _optimizer_to_gpu(opt):
    for state in opt.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(DEVICE)


def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _stage(dataset: str, split: str) -> Path:
    src = RAW_ROOT / dataset / f"{split}.txt"
    if not src.is_file():
        raise FileNotFoundError(
            f"Missing raw corpus: {src}\n"
            "This notebook does not download or tokenize data."
        )

    dst_dir = LOCAL_ROOT / dataset
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{split}.txt"

    if dst.is_file() and dst.stat().st_size == src.stat().st_size:
        print(
            f"LOCAL CACHE | {dataset}/{split} | "
            f"{dst.stat().st_size / 1024**2:.1f} MiB"
        )
        return dst

    print(
        f"STAGE | {dataset}/{split} | "
        f"{src.stat().st_size / 1024**2:.1f} MiB -> local SSD"
    )

    tmp = Path(str(dst) + ".tmp")
    tmp.unlink(missing_ok=True)
    shutil.copyfile(src, tmp)

    if tmp.stat().st_size != src.stat().st_size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Incomplete local copy: {src}")

    os.replace(tmp, dst)
    return dst


# -----------------------------------------------------------------------------
# PRE-FLIGHT MODEL IDENTITY
# -----------------------------------------------------------------------------

cleanup_cuda()
_seed_all(MODEL_SEED)

_preflight_model = CTWModel().to(DEVICE)

PARAM_COUNT = sum(p.numel() for p in _preflight_model.parameters())
TRAINABLE_COUNT = sum(
    p.numel() for p in _preflight_model.parameters() if p.requires_grad
)

if PARAM_COUNT != EXPECTED_PARAMS:
    del _preflight_model
    cleanup_cuda()
    raise RuntimeError(
        f"Wrong TransWave model: got {PARAM_COUNT:,} parameters; "
        f"expected {EXPECTED_PARAMS:,}."
    )

print("=" * 120)
print(RELEASE_NAME)
print(f"GPU                     : {GPU_NAME}")
print(f"VRAM                    : {GPU_TOTAL_GB:.2f} GiB")
print(f"architecture            : {ARCH}")
print(f"parameters              : {PARAM_COUNT:,}")
print(f"trainable               : {TRAINABLE_COUNT:,}")
print(f"context                 : {RAW_BYTES_PER_SEQUENCE:,} raw bytes")
print(f"training exposure       : {TRAIN_TOKENS:,} raw-byte tokens")
print("offline tokenizer       : NONE")
print("learned lifting tokenizer: INTERNAL TO MODEL")
print("=" * 120)

# Strict architecture causality check before spending days training.
causality_test(_preflight_model)

del _preflight_model
cleanup_cuda()


# -----------------------------------------------------------------------------
# FIND MAXIMUM SAFE G4 BATCH
#
# We know B=176 was safe on the earlier 95.6-GiB G4. This fresh notebook
# verifies it on the current runtime, then searches upward/downward using
# complete forward+backward+AdamW steps. The probe model is discarded.
# -----------------------------------------------------------------------------

def _safe_probe(batch: int):
    row = full_step_probe(int(batch))
    if not row.get("fits", False):
        return False, row
    safe = float(row["free"]) >= VRAM_HEADROOM_GB
    return safe, row


def _choose_batch():
    # Reuse a completed plan only for the same architecture and GPU capacity.
    if BATCH_PLAN_FILE.is_file():
        try:
            cached = json.loads(BATCH_PLAN_FILE.read_text(encoding="utf-8"))
            if (
                cached.get("architecture") == ARCH
                and abs(float(cached.get("gpu_total_gib", -1)) - GPU_TOTAL_GB) < 0.5
                and int(cached.get("selected_batch", 0)) >= 1
            ):
                print(
                    "BATCH PLAN CACHE | "
                    f"B={cached['selected_batch']} | "
                    f"free={cached['selected_free_gib']:.2f} GiB"
                )
                return int(cached["selected_batch"])
        except Exception:
            pass

    tested = {}
    start = min(BATCH_SEARCH_START, BATCH_SEARCH_CEILING)

    def test(b):
        b = int(b)
        if b not in tested:
            safe, row = _safe_probe(b)
            tested[b] = {"safe": bool(safe), **row}
            if row.get("fits", False):
                print(
                    f"PROBE B={b:<3d} | "
                    f"reserved={row['peak_reserved']:.2f} GiB | "
                    f"free={row['free']:.2f} GiB | "
                    f"{'SAFE' if safe else 'HEADROOM LIMIT'}"
                )
            else:
                print(f"PROBE B={b:<3d} | OOM")
        return tested[b]

    first = test(start)

    if first["safe"]:
        low_safe = start
        high_unsafe = None

        b = start + BATCH_SEARCH_STEP
        while b <= BATCH_SEARCH_CEILING:
            row = test(b)
            if row["safe"]:
                low_safe = b
                b += BATCH_SEARCH_STEP
            else:
                high_unsafe = b
                break

        if high_unsafe is None:
            selected = low_safe
        else:
            lo = low_safe + 1
            hi = high_unsafe - 1
            selected = low_safe

            while lo <= hi:
                mid = (lo + hi) // 2
                row = test(mid)
                if row["safe"]:
                    selected = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
    else:
        high_unsafe = start
        b = max(1, start - BATCH_SEARCH_STEP)
        low_safe = None

        while b >= 1:
            row = test(b)
            if row["safe"]:
                low_safe = b
                break
            if b == 1:
                break
            b = max(1, b - BATCH_SEARCH_STEP)

        if low_safe is None:
            raise RuntimeError("No safe physical batch found, including B=1.")

        lo = low_safe + 1
        hi = high_unsafe - 1
        selected = low_safe

        while lo <= hi:
            mid = (lo + hi) // 2
            row = test(mid)
            if row["safe"]:
                selected = mid
                lo = mid + 1
            else:
                hi = mid - 1

    selected_row = test(selected)

    plan = {
        "architecture": ARCH,
        "gpu": GPU_NAME,
        "gpu_total_gib": GPU_TOTAL_GB,
        "required_headroom_gib": VRAM_HEADROOM_GB,
        "selected_batch": int(selected),
        "selected_peak_allocated_gib": float(selected_row["peak_alloc"]),
        "selected_peak_reserved_gib": float(selected_row["peak_reserved"]),
        "selected_free_gib": float(selected_row["free"]),
        "tested": {str(k): v for k, v in sorted(tested.items())},
    }

    _atomic_json(plan, BATCH_PLAN_FILE)

    print(
        f"SELECTED PHYSICAL BATCH = {selected} | "
        f"free headroom={selected_row['free']:.2f} GiB"
    )

    return int(selected)


PHYSICAL_BATCH = _choose_batch()


# -----------------------------------------------------------------------------
# STAGE RAW DATA AND CREATE NON-OVERLAPPING 32K MEMMAP BLOCKS
# -----------------------------------------------------------------------------

train_mem = []
train_blocks = []
valid_mem = []
valid_blocks = []

manifest = {"train": {}, "validation": {}}

for ds in DATASETS:
    train_path = _stage(ds, "train")
    valid_path = _stage(ds, "validation")

    tm = np.memmap(train_path, dtype=np.uint8, mode="r")
    vm = np.memmap(valid_path, dtype=np.uint8, mode="r")

    nt = len(tm) // RAW_BYTES_PER_SEQUENCE
    nv = len(vm) // RAW_BYTES_PER_SEQUENCE

    if nt < 1:
        raise RuntimeError(f"No full training contexts for {ds}.")
    if nv < VAL_WINDOWS_PER_DATASET:
        raise RuntimeError(
            f"Need at least {VAL_WINDOWS_PER_DATASET} validation contexts for {ds}; "
            f"found {nv}."
        )

    tb = tm[: nt * RAW_BYTES_PER_SEQUENCE].reshape(
        nt, RAW_BYTES_PER_SEQUENCE
    )
    vb = vm[: nv * RAW_BYTES_PER_SEQUENCE].reshape(
        nv, RAW_BYTES_PER_SEQUENCE
    )

    train_mem.append(tm)
    train_blocks.append(tb)
    valid_mem.append(vm)
    valid_blocks.append(vb)

    manifest["train"][ds] = {
        "source": str(RAW_ROOT / ds / "train.txt"),
        "raw_bytes": int(len(tm)),
        "full_contexts": int(nt),
    }
    manifest["validation"][ds] = {
        "source": str(RAW_ROOT / ds / "validation.txt"),
        "raw_bytes": int(len(vm)),
        "full_contexts": int(nv),
    }

_atomic_json(manifest, DATA_MANIFEST)

counts = np.asarray(
    [len(x) for x in train_blocks],
    dtype=np.int64,
)

starts = np.concatenate(
    ([0], np.cumsum(counts)[:-1])
)

ends = np.cumsum(counts)

TOTAL_BLOCKS = int(ends[-1])

FULL_SEQUENCES_NEEDED, FINAL_PARTIAL = divmod(
    TRAIN_TOKENS,
    RAW_BYTES_PER_SEQUENCE,
)

assert FULL_SEQUENCES_NEEDED == 137_329
assert FINAL_PARTIAL == 0


# -----------------------------------------------------------------------------
# DETERMINISTIC SHUFFLED PASSES THROUGH ALL AVAILABLE 32K CONTEXTS
# -----------------------------------------------------------------------------

_perm_epoch = None
_perm_cache = None


def _epoch_permutation(epoch: int):
    global _perm_epoch, _perm_cache

    if _perm_epoch != epoch or _perm_cache is None:
        _perm_cache = np.random.default_rng(
            ORDER_SEED + epoch
        ).permutation(TOTAL_BLOCKS)
        _perm_epoch = epoch

    return _perm_cache


def _global_ids_for_cursor(cursor: int, count: int):
    pieces = []
    remaining = int(count)
    c = int(cursor)

    while remaining:
        epoch = c // TOTAL_BLOCKS
        offset = c % TOTAL_BLOCKS

        perm = _epoch_permutation(epoch)

        take = min(
            remaining,
            TOTAL_BLOCKS - offset,
        )

        pieces.append(
            perm[offset : offset + take]
        )

        c += take
        remaining -= take

    return np.concatenate(pieces).astype(
        np.int64,
        copy=False,
    )


def _cpu_batch(gids):
    arr = np.empty(
        (
            len(gids),
            RAW_BYTES_PER_SEQUENCE,
        ),
        dtype=np.uint8,
    )

    dsi = np.searchsorted(
        ends,
        gids,
        side="right",
    )

    for di in range(len(DATASETS)):
        rows = np.flatnonzero(dsi == di)

        if rows.size:
            local_ids = gids[rows] - starts[di]
            arr[rows] = np.asarray(
                train_blocks[di][local_ids],
                dtype=np.uint8,
            )

    return torch.from_numpy(
        np.ascontiguousarray(arr)
    ).pin_memory()


def _lr_for_tokens(tokens_seen: int):
    warmup_tokens = max(
        RAW_BYTES_PER_SEQUENCE,
        int(TRAIN_TOKENS * WARMUP_FRACTION),
    )

    if tokens_seen < warmup_tokens:
        return (
            BASE_LR
            * max(
                RAW_BYTES_PER_SEQUENCE,
                tokens_seen + RAW_BYTES_PER_SEQUENCE,
            )
            / warmup_tokens
        )

    progress = (
        tokens_seen - warmup_tokens
    ) / max(
        1,
        TRAIN_TOKENS - warmup_tokens,
    )

    progress = min(
        1.0,
        max(
            0.0,
            progress,
        ),
    )

    return (
        MIN_LR
        + (BASE_LR - MIN_LR)
        * 0.5
        * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )
    )


# -----------------------------------------------------------------------------
# PERMANENT TRAINING PROTOCOL
# -----------------------------------------------------------------------------

protocol = {
    "release_name": RELEASE_NAME,
    "architecture": ARCH,
    "objective": "exact causal next-byte likelihood",
    "training_token_definition": "one raw byte = one model token",
    "training_tokens": TRAIN_TOKENS,
    "full_contexts_required": int(FULL_SEQUENCES_NEEDED),
    "raw_bytes_per_context": RAW_BYTES_PER_SEQUENCE,
    "datasets": list(DATASETS),
    "sampling": (
        "all available non-overlapping 32768-byte contexts are shuffled "
        "without replacement per pass; passes repeat until 4.5B exposure"
    ),
    "parameter_count": PARAM_COUNT,
    "base_lr": BASE_LR,
    "min_lr": MIN_LR,
    "warmup_fraction": WARMUP_FRACTION,
    "weight_decay": WEIGHT_DECAY,
    "adam_betas": list(ADAM_BETAS),
    "adam_eps": ADAM_EPS,
    "grad_clip": GRAD_CLIP,
    "model_seed": MODEL_SEED,
    "data_order_seed": ORDER_SEED,
    "physical_batch_initial": PHYSICAL_BATCH,
    "checkpoint_every_steps": SAVE_EVERY_STEPS,
    "gpu": GPU_NAME,
    "gpu_total_gib": GPU_TOTAL_GB,
    "torch": torch.__version__,
}

fingerprint = hashlib.sha256(
    json.dumps(
        protocol,
        sort_keys=True,
    ).encode()
).hexdigest()

protocol["fingerprint"] = fingerprint
_atomic_json(protocol, PROTOCOL)


# -----------------------------------------------------------------------------
# ACTUAL MODEL + OPTIMIZER + RESUME
# -----------------------------------------------------------------------------

cleanup_cuda()
_seed_all(MODEL_SEED)

model = CTWModel().to(DEVICE)
optimizer, FUSED_ADAMW = make_optimizer(model)

actual_params = sum(
    p.numel() for p in model.parameters()
)

if actual_params != EXPECTED_PARAMS:
    raise RuntimeError(
        f"Parameter count changed before training: "
        f"{actual_params:,} != {EXPECTED_PARAMS:,}"
    )

seq_cursor = 0
tokens_seen = 0
step = 0
history = []
prior_seconds = 0.0

if LATEST.is_file():
    state = torch.load(
        LATEST,
        map_location="cpu",
        weights_only=False,
    )

    if state.get("fingerprint") != fingerprint:
        raise RuntimeError(
            "Existing recovery checkpoint belongs to a different run."
        )

    model.load_state_dict(
        state["model_state"],
        strict=True,
    )

    optimizer.load_state_dict(
        state["optimizer_state"]
    )

    _optimizer_to_gpu(
        optimizer
    )

    seq_cursor = int(
        state["seq_cursor"]
    )

    tokens_seen = int(
        state["tokens_seen"]
    )

    step = int(
        state["step"]
    )

    PHYSICAL_BATCH = int(
        state.get(
            "physical_batch",
            PHYSICAL_BATCH,
        )
    )

    history = list(
        state.get(
            "history",
            [],
        )
    )

    prior_seconds = float(
        state.get(
            "training_seconds",
            0.0,
        )
    )

    del state

    print(
        f"RESUME | step={step} | "
        f"tokens={tokens_seen:,}/{TRAIN_TOKENS:,} | "
        f"B={PHYSICAL_BATCH}"
    )


session_start = time.perf_counter()


def _elapsed():
    return (
        prior_seconds
        + time.perf_counter()
        - session_start
    )


def _save_recovery():
    _atomic_torch_save(
        {
            "fingerprint": fingerprint,
            "protocol": protocol,
            "seq_cursor": int(seq_cursor),
            "tokens_seen": int(tokens_seen),
            "step": int(step),
            "physical_batch": int(PHYSICAL_BATCH),
            "training_seconds": float(_elapsed()),
            "history": history,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        LATEST,
    )

    _atomic_json(
        history,
        HISTORY_FILE,
    )


# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------

print("=" * 120)
print("4.5B PRETRAINING START")
print(f"release name            : {RELEASE_NAME}")
print(f"available train contexts: {TOTAL_BLOCKS:,}")
print(f"required train contexts : {FULL_SEQUENCES_NEEDED:,}")
print(f"equivalent corpus passes: {FULL_SEQUENCES_NEEDED / TOTAL_BLOCKS:.3f}")
print(f"physical batch          : {PHYSICAL_BATCH}")
print(f"planned optimizer steps : {math.ceil((FULL_SEQUENCES_NEEDED - seq_cursor) / PHYSICAL_BATCH):,}")
print(f"recovery checkpoint     : every {SAVE_EVERY_STEPS} steps")
print(f"Drive run directory     : {RUN_ROOT}")
print("=" * 120)

model.train()
torch.cuda.reset_peak_memory_stats()

try:
    while seq_cursor < FULL_SEQUENCES_NEEDED:
        this_batch = min(
            PHYSICAL_BATCH,
            FULL_SEQUENCES_NEEDED - seq_cursor,
        )

        gids = _global_ids_for_cursor(
            seq_cursor,
            this_batch,
        )

        cpu = None
        ids = None
        loss = None

        try:
            cpu = _cpu_batch(
                gids
            )

            ids = cpu.to(
                DEVICE,
                dtype=torch.long,
                non_blocking=True,
            )

            lr = _lr_for_tokens(
                tokens_seen
            )

            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(
                set_to_none=True
            )

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.autocast(
                "cuda",
                dtype=AMP_DTYPE,
            ):
                loss = model.nll(
                    ids,
                    checkpoint_layers=True,
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

            torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            step_tokens = int(
                this_batch
                * RAW_BYTES_PER_SEQUENCE
            )

            seq_cursor += int(
                this_batch
            )

            tokens_seen += step_tokens
            step += 1

            nll = float(
                loss.detach()
            )

            peak_alloc = float(
                torch.cuda.max_memory_allocated()
                / 1024**3
            )

            peak_reserved = float(
                torch.cuda.max_memory_reserved()
                / 1024**3
            )

            row = {
                "step": int(step),
                "tokens_seen": int(tokens_seen),
                "batch": int(this_batch),
                "nll": nll,
                "ppl": float(
                    math.exp(
                        min(
                            nll,
                            20.0,
                        )
                    )
                ),
                "bpb": float(
                    nll
                    / math.log(
                        2.0
                    )
                ),
                "lr": float(lr),
                "seconds": float(dt),
                "tokens_per_second": float(
                    step_tokens
                    / max(
                        dt,
                        1e-9,
                    )
                ),
                "peak_allocated_gib": peak_alloc,
                "peak_reserved_gib": peak_reserved,
            }

            history.append(
                row
            )

            eta_s = (
                max(
                    0,
                    TRAIN_TOKENS - tokens_seen,
                )
                / max(
                    row["tokens_per_second"],
                    1e-9,
                )
            )

            print(
                f"step={step:4d} | "
                f"tokens={tokens_seen:>13,d}/{TRAIN_TOKENS:,} | "
                f"B={this_batch:<3d} | "
                f"NLL={nll:.4f} | "
                f"PPL={row['ppl']:.3f} | "
                f"lr={lr:.2e} | "
                f"{row['tokens_per_second']:,.0f} tok/s | "
                f"VRAM={peak_alloc:.2f}/{peak_reserved:.2f} GiB | "
                f"ETA={eta_s / 3600:.2f} h"
            )

            del (
                cpu,
                ids,
                loss,
                gids,
            )

            if (
                step
                % SAVE_EVERY_STEPS
                ==
                0
            ):
                _save_recovery()
                print(
                    f"RECOVERY SAVED | "
                    f"{tokens_seen:,} tokens"
                )

        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(
                set_to_none=True
            )

            cpu = None
            ids = None
            loss = None
            gids = None

            cleanup_cuda()

            new_batch = max(
                1,
                PHYSICAL_BATCH - max(
                    1,
                    PHYSICAL_BATCH // 16,
                ),
            )

            if new_batch >= PHYSICAL_BATCH:
                new_batch = PHYSICAL_BATCH - 1

            if new_batch < 1:
                raise RuntimeError(
                    "OOM even at physical batch 1."
                )

            print(
                f"RUNTIME OOM RECOVERY | "
                f"B {PHYSICAL_BATCH} -> {new_batch} | "
                "same data will be retried"
            )

            PHYSICAL_BATCH = int(
                new_batch
            )

            _save_recovery()

except BaseException:
    try:
        _save_recovery()
        print(
            "EMERGENCY RECOVERY CHECKPOINT SAVED TO DRIVE."
        )
    except Exception as save_error:
        print(
            "Emergency save failed:",
            save_error,
        )
    raise


assert seq_cursor == FULL_SEQUENCES_NEEDED
assert tokens_seen == TRAIN_TOKENS


# -----------------------------------------------------------------------------
# SMALL FIXED HELD-OUT VALIDATION
# -----------------------------------------------------------------------------

@torch.no_grad()
def _final_validation():
    model.eval()

    per_dataset = {}
    total_loss = 0.0
    total_tokens = 0

    val_batch = max(
        1,
        min(
            PHYSICAL_BATCH,
            VAL_WINDOWS_PER_DATASET,
        ),
    )

    for di, ds in enumerate(DATASETS):
        ds_loss = 0.0
        ds_tokens = 0

        for start in range(
            0,
            VAL_WINDOWS_PER_DATASET,
            val_batch,
        ):
            end = min(
                VAL_WINDOWS_PER_DATASET,
                start + val_batch,
            )

            arr = np.asarray(
                valid_blocks[di][start:end],
                dtype=np.uint8,
            )

            cpu = torch.from_numpy(
                np.ascontiguousarray(arr)
            ).pin_memory()

            ids = cpu.to(
                DEVICE,
                dtype=torch.long,
                non_blocking=True,
            )

            with torch.autocast(
                "cuda",
                dtype=AMP_DTYPE,
            ):
                nll = model.nll(
                    ids,
                    checkpoint_layers=False,
                )

            nt = int(
                (end - start)
                * RAW_BYTES_PER_SEQUENCE
            )

            ds_loss += float(
                nll.detach()
            ) * nt

            ds_tokens += nt

            del (
                cpu,
                ids,
                nll,
                arr,
            )

        mean_nll = (
            ds_loss
            / ds_tokens
        )

        per_dataset[ds] = {
            "nll": float(mean_nll),
            "ppl": float(
                math.exp(
                    min(
                        mean_nll,
                        20.0,
                    )
                )
            ),
            "bpb": float(
                mean_nll
                / math.log(
                    2.0
                )
            ),
            "tokens": int(ds_tokens),
        }

        total_loss += ds_loss
        total_tokens += ds_tokens

    overall_nll = (
        total_loss
        / total_tokens
    )

    model.train()

    return {
        "overall": {
            "nll": float(overall_nll),
            "ppl": float(
                math.exp(
                    min(
                        overall_nll,
                        20.0,
                    )
                )
            ),
            "bpb": float(
                overall_nll
                / math.log(
                    2.0
                )
            ),
            "tokens": int(total_tokens),
        },
        "per_dataset": per_dataset,
    }


print("RUNNING FINAL HELD-OUT CHECK...")
validation = _final_validation()

print(
    "FINAL | "
    f"NLL={validation['overall']['nll']:.6f} | "
    f"PPL={validation['overall']['ppl']:.4f} | "
    f"BPB={validation['overall']['bpb']:.4f}"
)


# -----------------------------------------------------------------------------
# FINAL CHECKPOINT
# -----------------------------------------------------------------------------

result = {
    "status": "complete",
    "release_name": RELEASE_NAME,
    "fingerprint": fingerprint,
    "architecture": ARCH,
    "training_tokens": TRAIN_TOKENS,
    "token_definition": "one raw byte",
    "steps": int(step),
    "physical_batch_final": int(PHYSICAL_BATCH),
    "parameter_count": int(PARAM_COUNT),
    "trainable_parameter_count": int(TRAINABLE_COUNT),
    "fused_adamw": bool(FUSED_ADAMW),
    "training_seconds": float(_elapsed()),
    "validation": validation,
    "peak_allocated_gib": float(
        torch.cuda.max_memory_allocated()
        / 1024**3
    ),
    "peak_reserved_gib": float(
        torch.cuda.max_memory_reserved()
        / 1024**3
    ),
}

_atomic_torch_save(
    {
        **result,
        "protocol": protocol,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    },
    FINAL,
)

_atomic_json(
    result,
    RESULT,
)

_atomic_json(
    history,
    HISTORY_FILE,
)


# -----------------------------------------------------------------------------
# HF WEIGHT EXPORT
#
# The complete HF repository will be assembled from this directory after the
# run. No external tokenizer is required; the model consumes raw byte IDs.
# -----------------------------------------------------------------------------

weight_state = {
    key: value.detach().cpu().contiguous()
    for key, value
    in model.state_dict().items()
}

weight_file = None

try:
    from safetensors.torch import save_file

    weight_path = (
        HF_EXPORT
        / "model.safetensors"
    )

    save_file(
        weight_state,
        str(
            weight_path
        ),
    )

    weight_file = (
        weight_path.name
    )

except Exception:
    weight_path = (
        HF_EXPORT
        / "pytorch_model.bin"
    )

    torch.save(
        weight_state,
        weight_path,
    )

    weight_file = (
        weight_path.name
    )

del weight_state


_atomic_json(
    {
        "release_name": RELEASE_NAME,
        "architecture": ARCH,
        "parameter_count": PARAM_COUNT,
        "training_tokens": TRAIN_TOKENS,
        "token_definition": "raw byte IDs 0..255",
        "context_length_bytes": RAW_BYTES_PER_SEQUENCE,
        "weight_file": weight_file,
        "validation": validation,
    },
    HF_EXPORT
    / "export_manifest.json",
)


LATEST.unlink(
    missing_ok=True
)


print("=" * 120)
print("4.5B PRETRAINING COMPLETE")
print(f"release name            : {RELEASE_NAME}")
print(f"tokens                  : {tokens_seen:,}")
print(f"steps                   : {step}")
print(
    "held-out              : "
    f"NLL={validation['overall']['nll']:.6f} | "
    f"PPL={validation['overall']['ppl']:.4f} | "
    f"BPB={validation['overall']['bpb']:.4f}"
)
print(f"full recovery checkpoint: {FINAL}")
print(f"HF weight export        : {HF_EXPORT / weight_file}")
print(f"HF export directory     : {HF_EXPORT}")
print("=" * 120)
