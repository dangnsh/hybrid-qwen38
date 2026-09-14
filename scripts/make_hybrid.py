#!/usr/bin/env python3
"""Build the hybrid model directory.

Output: HYBRID_DIR (default ~/.cache/huggingface/hybrid-qwen38) containing
  - relative symlinks to every file in the NVIDIA snapshot (except the index)
  - hybrid-mtp-bf16.safetensors: the fused MTP tensors copied from RadixArk
  - a patched model.safetensors.index.json: NVIDIA's map minus its per-expert
    mtp.* keys, with mtp.* rerouted to the new shard

Run this on EVERY node that will serve the model, with the same HF cache layout.
CPU-only; touches no GPU. Requires: pip install safetensors torch

Env overrides:
  HYBRID_DIR   output directory
  NV_REPO_ID   main-weights repo  (default nvidia/Qwen3.8-Flash-Next-NVFP4)
  RA_REPO_ID   fused-MTP source   (default RadixArk/Qwen3.8-Flash-Next-NVFP4)
"""
import glob
import json
import os
import sys

from safetensors import safe_open
from safetensors.torch import save_file

MTP_FILE = "hybrid-mtp-bf16.safetensors"


def snapshot(repo_id: str) -> str:
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    pat = f"{hub}/models--{repo_id.replace('/', '--')}/snapshots/*"
    hits = sorted(glob.glob(pat), key=os.path.getmtime)
    if not hits:
        sys.exit(f"no local snapshot for {repo_id}. Run: huggingface-cli download {repo_id}")
    return hits[-1]


def main() -> None:
    nv_repo = os.environ.get("NV_REPO_ID", "nvidia/Qwen3.8-Flash-Next-NVFP4")
    ra_repo = os.environ.get("RA_REPO_ID", "RadixArk/Qwen3.8-Flash-Next-NVFP4")
    out = os.path.expanduser(os.environ.get("HYBRID_DIR", "~/.cache/huggingface/hybrid-qwen38"))

    nv, ra = snapshot(nv_repo), snapshot(ra_repo)
    nv_map = json.load(open(f"{nv}/model.safetensors.index.json"))["weight_map"]
    ra_map = json.load(open(f"{ra}/model.safetensors.index.json"))["weight_map"]

    ra_mtp = {k: v for k, v in ra_map.items() if k.startswith("mtp.")}
    nv_mtp = [k for k in nv_map if k.startswith("mtp.")]
    print(f"RA fused-mtp keys={len(ra_mtp)} | NV mtp keys={len(nv_mtp)}")
    if not ra_mtp:
        sys.exit("RadixArk snapshot has no fused mtp.* keys; wrong checkpoint?")

    # 1. lift the fused MTP tensors out of the RadixArk shards
    tensors = {}
    for fn in sorted(set(ra_mtp.values())):
        with safe_open(f"{ra}/{fn}", framework="pt") as f:
            for k in f.keys():
                if k in ra_mtp:
                    tensors[k] = f.get_tensor(k)
    missing = [k for k in ra_mtp if k not in tensors]
    if missing:
        sys.exit(f"missing {len(missing)} fused mtp tensors, e.g. {missing[:3]}")

    # 2. mirror the NVIDIA snapshot with RELATIVE symlinks (absolute host paths
    #    are invisible inside the container where the cache is remounted)
    os.makedirs(out, exist_ok=True)
    for name in sorted(os.listdir(nv)):
        if name == "model.safetensors.index.json":
            continue
        dst = os.path.join(out, name)
        if os.path.islink(dst) or os.path.exists(dst):
            continue
        os.symlink(os.path.relpath(os.path.join(nv, name), out), dst)

    # 3. write the fused MTP shard + rerouted index
    save_file(tensors, f"{out}/{MTP_FILE}", metadata={"format": "pt"})
    new_map = {k: v for k, v in nv_map.items() if not k.startswith("mtp.")}
    for k in ra_mtp:
        new_map[k] = MTP_FILE
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    json.dump({"metadata": {"total_size": total}, "weight_map": new_map},
              open(f"{out}/model.safetensors.index.json", "w"))

    dropped = len(nv_mtp) - sum(1 for k in ra_mtp if k in nv_map)
    print(f"WROTE {out}: {len(tensors)} mtp tensors ({total / 1e9:.2f} GB), "
          f"index keys={len(new_map)}, dropped NV-only mtp keys={dropped}")
    links = sum(1 for f in os.listdir(out) if os.path.islink(os.path.join(out, f)))
    print(f"symlinks={links}, files={len(os.listdir(out))}")
    print("next: patch the MTP loader (scripts/patch_mtp.py), then launch with HYBRID_MTP=1")


if __name__ == "__main__":
    main()
