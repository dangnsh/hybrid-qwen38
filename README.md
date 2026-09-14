# Hybrid Qwen3.8-Flash-Next: NVIDIA weights + fused MTP

Serve NVIDIA's official `nvidia/Qwen3.8-Flash-Next-NVFP4` checkpoint on a
2-node DGX Spark cluster **with MTP speculative decoding still on**, by pairing
those weights with the fused BF16 MTP tensors from the RadixArk checkpoint.

If you have two GB10 boxes and want the NVIDIA calibration without losing half
your decode speed, this is the workaround. This repo is the full setup guide:
what to download, what to patch, how to launch, how to verify, and the failure
modes we hit so you can skip the debugging.

The short version of the setup:

```
download both checkpoints -> make_hybrid.py -> patch_mtp.py -> launch worker+head -> qa.py
```

Everything else in this file is those five steps with their explanations.

## The problem

`nvidia/Qwen3.8-Flash-Next-NVFP4` and `RadixArk/Qwen3.8-Flash-Next-NVFP4` are the
same architecture, quantized from the same base by the same tool (ModelOpt PTQ).
NVIDIA's version ships better calibration data and a stronger eval table. Both
are about 132 GB, and both fit two Sparks.

The MTP head is where they diverge:

| | MTP expert layout |
|---|---|
| `RadixArk/...` | fused `w13` / `w2` (+ scale) |
| `nvidia/...` | per-expert tensors + NVFP4 scale keys |

The vLLM day-0 image for this model (`vllm/vllm-openai` with the Flash-Next
model definitions) builds a **fused** MTP draft module. Point it at NVIDIA's
checkpoint with speculative decoding enabled and it dies mid-load:

```
AttributeError: Layer mtp.layers.48.mlp.experts has no parameter
'w2_weight_scale_inv' for checkpoint weight
'mtp.layers.48.mlp.experts.0.down_proj.weight_scale_inv'
```

Turn MTP off and NVIDIA loads fine. On our pair that costs a lot:

| Config | Decode (c1, prose) |
|---|---|
| RadixArk + MTP3 (baseline) | 48.2 / 52.3 / 52.6 tok/s |
| NVIDIA, MTP off | 25.3 tok/s |
| **Hybrid (this repo), MTP3** | **48.0 / 52.8 / 54.1 tok/s** |

So the choice as shipped is "NVIDIA weights at half speed" or "RadixArk at full
speed". We wanted both.

## The approach

Two artifacts, no image rebuild:

1. **A hybrid model directory.** Symlinks to every NVIDIA shard, plus one new
   safetensors file holding the 31 fused MTP tensors lifted out of the RadixArk
   checkpoint (about 5.2 GB in BF16). A patched index drops NVIDIA's per-expert
   MTP keys and points `mtp.*` at the new shard. Disk cost is the MTP shard
   alone; the rest are symlinks.
2. **A bind-mounted patch to the MTP loader.** It skips per-expert MTP checkpoint
   names, so the draft module gets its fused tensors from the hybrid shard and
   never tries to match NVIDIA's expert keys.

The main weights are 100% NVIDIA. Only the drafter comes from RadixArk, and a
speculative drafter only has to be a decent guesser: whatever it proposes is
verified by the NVIDIA target model. That's the reason this pairing is sane.

## Requirements

- 2x NVIDIA DGX Spark (GB10, SM121), 121 GiB unified memory each
- vLLM image with Qwen3.8-Flash-Next support + SM121 patches (our build is
  `qwen38-flash-dgx:v3-blazux`; adjust `IMAGE` if you use a different one)
- RoCE/IB cabled between the two nodes, one port pair known to be up
- ~140 GiB free per node for the NVIDIA checkpoint, plus 5 GiB for the MTP shard
- `safetensors` + `torch` available in a Python env on each node (CPU only for
  the build step; nothing here touches a GPU)

Both checkpoints come from Hugging Face:

```bash
huggingface-cli download nvidia/Qwen3.8-Flash-Next-NVFP4
huggingface-cli download RadixArk/Qwen3.8-Flash-Next-NVFP4
```

## Usage

### 1. Extract the loader files from your image

The patch targets files inside the serving image. Pull the ones you need out to
the launch directory:

```bash
CID=$(docker create IMAGE)
docker cp $CID:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/mtp.py ./mtp_patched.py
docker rm $CID
```

`ple_layer.py` from the same directory also gets bind-mounted (FP32 PLE blows up
on this hardware; the FP8 force flag plus a patched copy is the standing
workaround, unrelated to the hybrid body). See `docs/notes.md` for details.

### 2. Build the hybrid body (on both nodes)

```bash
python3 scripts/make_hybrid.py
```

It finds both checkpoints in your HF cache, writes
`~/.cache/huggingface/hybrid-qwen38`, and prints what it did. Expect:

```
RA mtp keys=31 in 1 shards | NV mtp keys=3101
extracted=31 missing=[] | NV-only keys kept out=3072
WROTE .../hybrid-qwen38: 31 mtp tensors (5.21 GB), index keys=296475
```

Run it on each node. Keep the directory layout identical on both, or rank 1 will
fail to resolve the index.

### 3. Patch the MTP loader

```bash
python3 scripts/patch_mtp.py ./mtp_patched.py
```

Five lines of runtime logic: if a checkpoint weight name looks like
`.experts.<N>.`, return `None` and let the fused tensors win.

### 4. Launch

Worker first, then head:

```bash
export HEAD_IP=<head-node-ip> WORKER_IP=<worker-node-ip>
export IFACE=<iface> IB_HCA=<roce-device>

HYBRID_MTP=1 DRAFT_VOCAB=0 ./scripts/launch.sh 1   # on the worker
HYBRID_MTP=1 DRAFT_VOCAB=0 ./scripts/launch.sh 0   # on the head, serves :8000
```

Boot takes about 11 minutes on our pair: roughly 8 to stream 65 GiB of weights
per rank, then KV/mamba page setup and CUDA graph capture. Do not conclude
anything from port 8000 before that.

`MTP_OFF=1` runs the plain NVIDIA body (no speculation) if you want to compare.

## Verify

```bash
curl -s localhost:8000/v1/models | jq -r '.data[0].root'
# /root/.cache/huggingface/hybrid-qwen38

python3 scripts/qa.py            # tools, math, loop stress, throughput
```

You should see `SpeculativeConfig(method='mtp', num_spec_tokens=3)` in the head
log, `Resolved architecture: Qwen3_8FlashNextMTP`, and zero `AttributeError`.

## Results on our hardware

Decode speed is the table at the top. Beyond that, the hybrid body passed our
usual gates, same as the RadixArk baseline:

- Vietnamese prose: 1,685 chars, 245 diacritics, no CJK leakage, 4-gram
  repetition rate 0.003
- Tool calling: 2 calls, correct names, valid JSON arguments
- Loop stress: three identical rounds, output lengths 138 / 139 / 135, no growth
- `17 * 19` still says 323

What we did **not** verify: any language-quality gain from NVIDIA's calibration.
Their eval table is better on IFBench and a few other public suites, but there
is no Vietnamese benchmark anywhere, and our own prose sample is one sample.
Honest read: we got the NVIDIA main weights, kept our speed, and gave up nothing
measurable. Whether it writes better Vietnamese than RadixArk is still an open
question we happen to be living with.

## Traps we hit, so you don't have to

- **Symlinks must be relative.** Absolute host paths look fine on the host and
  are invisible inside the container, where the cache is mounted at
  `/root/.cache`. The failure is a pydantic config error at engine init, which
  is a fun one to trace.
- **TP=2 restarts have to be coordinated.** If only one rank restarts, the other
  keeps an hour-old process and the pair never rejoins: the head sits at
  `world_size=2` waiting on the master port forever. Stop both, start the worker,
  wait ~25 s, start the head.
- **A big model needs a cold page cache.** If you just downloaded or rsync'd the
  checkpoint, drop caches on both nodes before booting (`sync; echo 3 | sudo tee
  /proc/sys/vm/drop_caches`). mmap'd shards pin that cache, available RAM
  collapses mid-load, and the worker dies with no traceback. vLLM just says
  "WorkerProc initialization failed". The line
  `Checkpoint size X GiB. Available RAM Y GiB` right before the death is the
  tell.
- **A heavy `docker build` on the worker node can starve it into an sshd
  failure**: ping answers, TCP 22 connects, no banner. A TP=2 brain is down even
  though its own container is fine.
- **Health 200 is not a passing test.** Send a real chat request.

## Files

```
scripts/make_hybrid.py   build the hybrid model dir (symlinks + fused MTP shard + index)
scripts/patch_mtp.py     patch the extracted MTP loader to skip per-expert keys
scripts/launch.sh        docker run for rank 0/1, HYBRID_MTP + MTP_OFF switches
scripts/qa.py            endpoint QA: tools, math, loop stress, throughput
docs/notes.md            full log layout, watchdog design, extra observations
```

## Credits

Model checkpoints: `nvidia/Qwen3.8-Flash-Next-NVFP4` (NVIDIA, ModelOpt PTQ) and
`RadixArk/Qwen3.8-Flash-Next-NVFP4` (RadixArk). Serving runtime: vLLM's day-0
Flash-Next image. This repo is the glue, and the bug reports that made the glue
necessary.
