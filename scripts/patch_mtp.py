#!/usr/bin/env python3
"""Patch an extracted vLLM MTP loader so it ignores per-expert MTP checkpoint keys.

Why: nvidia/Qwen3.8-Flash-Next-NVFP4 ships MTP experts as per-expert tensors with
NVFP4 scale keys. The image's MTP draft module is built for the FUSED layout
(w13/w2) and raises AttributeError mid-weight-load on those names. The hybrid
body supplies fused MTP tensors from RadixArk (scripts/make_hybrid.py); this
patch makes the loader skip the per-expert names instead of trying to map them.

Usage:
    python3 patch_mtp.py ./mtp_patched.py
Then bind-mount the result over the image's copy at container start (launch.sh
does this when HYBRID_MTP=1).

Idempotent: running it twice is a no-op.
"""
import re
import sys

PATCH_MARK = "hybrid-body patch"

# the weight-name remapper inside the standalone MTP draft model; it is the one
# function that turns checkpoint names into module parameters, so returning None
# here is the cleanest skip point.
ANCHOR = re.compile(r'(def [A-Za-z_]+\([^)]*name[^)]*\)[^:]*:\n)(?:[ \t]*("""[^"]*"""\n)|)')


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: patch_mtp.py <path-to-mtp_patched.py>")
    path = sys.argv[1]
    src = open(path).read()
    if PATCH_MARK in src:
        print("already patched")
        return

    inject = (
        "    # hybrid-body patch: nvidia/* ships MTP experts per-expert + NVFP4 scales;\n"
        "    # the draft module wants fused w13/w2 from the hybrid shard. Skip them.\n"
        "    import re as _re_hybrid\n"
        "    if _re_hybrid.search(r'\\.experts\\.\\d+\\.', name):\n"
        "        return None\n"
    )

    # find the docstring of the remapper and inject right after it; fall back to
    # injecting right after the def line
    m = ANCHOR.search(src)
    if not m:
        sys.exit("could not locate the MTP weight-name remapper; this patch is\n"
                 "pinned to the vLLM day-0 Flash-Next image's mtp.py layout. Open\n"
                 "the file and apply the 3-line skip inside the function that maps\n"
                 "checkpoint weight names to draft-module parameters.")
    insert_at = m.end()
    open(path, "w").write(src[:insert_at] + inject + src[insert_at:])
    print(f"patched {path}")
    print("verify: grep -n 'hybrid-body patch' " + path)


if __name__ == "__main__":
    main()
