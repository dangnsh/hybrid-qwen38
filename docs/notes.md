# Notes

Extras that would clutter the README. Read these when something behaves oddly.

## What the hybrid body changes, exactly

| Component | Source |
|---|---|
| Embeddings, attention, MoE experts, LM head, vision tower | NVIDIA checkpoint |
| MTP draft module (1 layer + its experts) | RadixArk fused BF16 tensors |
| Config / tokenizer / chat template | NVIDIA snapshot (symlinked) |

Only the drafter is borrowed. Speculative decoding verifies every drafted token
against the target model, so the drafter's provenance can affect speed, not
correctness. That's what makes this safer than it first looks.

Disk accounting: about 132 GB of real bytes (NVIDIA) + about 5.2 GB fused MTP
shard. The symlinks are free. Both nodes need the same layout; the container
mounts `~/.cache/huggingface` at the same path inside, which is why the
symlinks are relative.

## Weight-count sanity checks

NVIDIA index reports roughly 299,545 tensor keys, of which about 3,101 belong to
`mtp.*`, 333 to the vision tower, 131 to ngram tables. RadixArk packs the same
MTP module into 31 fused tensors. After building, the hybrid index should sit
near 296,475 keys (NVIDIA minus its per-expert MTP keys, plus 31). If your
numbers look different, the two checkpoints are not the same vintage; stop and
compare configs before launching.

## Failure modes, in the order we hit them

**1. pydantic config error at engine init, right after "Resolved architecture"**
The model directory symlinks are absolute host paths. Inside the container they
point nowhere. Fix: rebuild with relative links (current `make_hybrid.py` does).

**2. AttributeError on `w2_weight_scale_inv` mid-shard-load**
This is the headline bug; MTP on, NVIDIA checkpoint, unpatched loader. Either
`MTP_OFF=1` (slow but works) or apply the loader patch.

**3. "WorkerProc initialization failed" with no traceback, on a machine that
just downloaded the model**
Host RAM exhaustion, not a code error. mmap'd shards pin the page cache;
available RAM collapses during load and the worker gets SIGKILLed. Look for
`Checkpoint size X GiB. Available RAM Y GiB` in the log just before the death.
`launch.sh` drops caches now, and the new upstream recipe does too.

**4. Head rank hangs forever at `world_size=2` on the master port**
Rank 1 was never started, or is a stale process from an earlier run. TP=2 pairs
do not rejoin after a one-sided restart. Stop both, start the worker, wait ~25
s, start the head.

**5. sshd on the worker node stops answering (TCP connects, no banner; later
connection resets) while ping still works**
Something ate the node's CPU/memory (for us: a kernel-heavy `docker build`
running while the brain was serving). The TP=2 brain is down even though its own
container is healthy. Do not conclude the cluster is broken; check what is
competing on the worker.

**6. Health endpoint 200 but replies are empty or truncated**
Two common causes: the token budget was swallowed by a reasoning block (retry
with a bigger `max_tokens` or thinking off), or the served model is fine but the
client sends a field the server rejects. Test with `qa.py`, which prints the
last 20 chars of the reply text.

## Watchdog sketch

The pattern that saved this deployment twice: a cron job every 2 minutes, gated
on a flag file so it does nothing when no trial is running, requiring 2
consecutive failed ticks before acting, with a grace window for slow boots
(grace measured in wall-clock minutes, never in tick counts; the first version
of ours killed a perfectly healthy boot at 5 minutes).

Its recovery action is deliberately boring: `docker start` the existing
containers in worker-then-head order. It does not switch models or edit configs.
An auto-rollback that rewrites production is how you end up with a 3am surprise;
"reload whatever is configured" is how you end up with a boring log.

Dump the container logs *before* removing a dead container. We lost the stack
trace that explained failure mode 2 because the cleanup ran first, and re-derived
it an hour later.

## Fabric notes

On GB10 the NIC exposes two ports (`...np0`, `...np1`); often only one is cabled.
Pick the interface carrying your inter-node subnet, and set `NCCL_IB_HCA` to a
single exact device. Listing a port cabled to some other cluster produces NCCL
behavior that will ruin an afternoon. `ip -o -4 addr` plus `ibv_devinfo` tells
you the truth in ten seconds.

## What we would like from upstream

- A loader that accepts either MTP layout (per-expert or fused) so the hybrid
  trick becomes unnecessary, or NVIDIA shipping a fused variant.
- A published Vietnamese or multilingual eval for these checkpoints. The current
  eval table is English-centric, which limits how much we can claim here.

## Measuring decode speed the same way we did

Three prose prompts at concurrency 1, 256-512 tokens each, `temperature=0`, take
median completion_tokens/elapsed. Report TTFT separately. Recipe READMEs that
quote streaming benchmarks without TTFT tend to look better than the same stack
measured end-to-end.
