# python/sglang/kernels/ops/diffusion/triton/ltx2_rotary_selector.py

import os


VARIANT = os.environ.get("SGLANG_LTX2_ROTARY_VARIANT", "B0").upper()


if VARIANT == "B0":
    from sglang.kernels.ops.diffusion.triton.ltx2_rotary import (
        apply_ltx2_split_rotary_emb,
    )

elif VARIANT == "B1":
    from sglang.kernels.ops.diffusion.triton.ltx2_rotary_b1 import (
        apply_ltx2_split_rotary_emb,
    )

elif VARIANT == "B3":
    from sglang.kernels.ops.diffusion.triton.ltx2_rotary_b3 import (
        apply_ltx2_split_rotary_emb,
    )

else:
    raise RuntimeError(
        f"Unknown SGLANG_LTX2_ROTARY_VARIANT={VARIANT!r}; "
        "expected B0, B1 or B3"
    )


print(
    f"[LTX2 ROTARY] selected variant={VARIANT}",
    flush=True,
)
