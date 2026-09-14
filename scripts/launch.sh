#!/bin/bash
# Launch one rank of the hybrid Qwen3.8-Flash-Next body on a 2-node GB10 pair.
#
#   worker first:  HYBRID_MTP=1 ./launch.sh 1     (on the worker node)
#   head second :  HYBRID_MTP=1 ./launch.sh 0     (on the head node, serves :8000)
#
# Copy mtp_patched.py (and ple_layer_patched.py) next to this script first —
# see README step 1. Boot takes ~11 min per node on GB10; port 8000 answers
# nothing until then.
#
# Env (set these to YOUR fabric):
#   HEAD_IP WORKER_IP IFACE IB_HCA          inter-node wiring
#   IMAGE            serving image with Flash-Next support (default below)
#   MODEL            model dir (default: the hybrid dir built by make_hybrid.py)
#   HYBRID_MTP=1     bind-mount mtp_patched.py (required for the hybrid body)
#   MTP_OFF=1        disable speculative decoding (plain NVIDIA comparison run)
#   SEQS GMU         max-num-seqs / gpu-memory-utilization overrides
set -euo pipefail
RANK="${1:?usage: launch.sh <0|1>}"

IMAGE="${IMAGE:-qwen38-flash-dgx:v3-blazux}"
NAME="${NAME:-vllm-fn}"
MODEL="${MODEL:-$HOME/.cache/huggingface/hybrid-qwen38}"

HEAD_IP="${HEAD_IP:?set HEAD_IP to the rank-0 inter-node IP}"
WORKER_IP="${WORKER_IP:?set WORKER_IP to the rank-1 inter-node IP}"
IFACE="${IFACE:?set IFACE to the RoCE interface carrying the inter-node subnet}"
IB_HCA="${IB_HCA:?set IB_HCA to the RoCE device, e.g. =rocep1s0f0 (exact match)}"

MPORT=50000 PORT=8000
case "$RANK" in
  0) HOST_IP="$HEAD_IP";   EXTRA="--host 0.0.0.0 --port $PORT" ;;
  1) HOST_IP="$WORKER_IP"; EXTRA="--headless" ;;
  *) echo "rank must be 0 or 1"; exit 2 ;;
esac
SEQS="${SEQS:-8}"; GMU="${GMU:-0.72}"

if [ "${MTP_OFF:-0}" = "1" ]; then SPEC=""; else SPEC='--speculative-config {"method":"mtp","num_speculative_tokens":3}'; fi

HERE="$(cd "$(dirname "$0")" && pwd)"
MOUNTS=""
if [ "${HYBRID_MTP:-0}" = "1" ]; then
  test -f "$HERE/mtp_patched.py" || { echo "missing $HERE/mtp_patched.py (README step 1+3)"; exit 1; }
  MOUNTS="$MOUNTS -v $HERE/mtp_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/mtp.py:ro"
fi
if [ -f "$HERE/ple_layer_patched.py" ]; then
  MOUNTS="$MOUNTS -v $HERE/ple_layer_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro"
fi

# Page cache from a fresh download/rsync pins itself against mmap and collapses
# available RAM mid-load (worker dies, no traceback). Cheap insurance:
if command -v sudo >/dev/null && sudo -n true 2>/dev/null; then
  sync && echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null && echo "dropped page cache"
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
mkdir -p ~/.cache/vllm
docker run -d --name "$NAME" --gpus all --network host --ipc host \
  --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864 \
  --device /dev/infiniband:/dev/infiniband \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e GLOO_SOCKET_IFNAME="$IFACE" -e NCCL_SOCKET_IFNAME="$IFACE" -e TP_SOCKET_IFNAME="$IFACE" \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$IB_HCA" \
  -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_AUTO_DETECT=0 -e NCCL_DEBUG=WARN \
  -e PLE_FORCE_FP8=1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm:/root/.cache/vllm \
  $MOUNTS \
  "$IMAGE" \
  "$MODEL" \
    --served-model-name "${SERVED_NAME:-qwen3.8-flash-next}" \
    --distributed-executor-backend mp \
    --nnodes 2 --node-rank "$RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --all2all-backend allgather_reducescatter \
    --load-format safetensors --safetensors-load-strategy lazy \
    --max-model-len "${MAX_MODEL_LEN:-262144}" \
    --max-num-seqs "$SEQS" \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization "$GMU" \
    --enable-chunked-prefill \
    $SPEC \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    $EXTRA
echo "launched $NAME rank=$RANK host=$HOST_IP"
sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || { echo "WARNING: $NAME not running; docker logs $NAME" >&2; exit 1; }
