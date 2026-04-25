"""
hickle bug-hunt (Step 7).

We probe edge cases that the unit tests don't really exercise. Goal is
NOT to break USABLE — but to honestly report what survived and what
didn't. Each probe is wrapped so one failure doesn't mask the others.
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback

import numpy as np
import hickle


RESULTS = []


def probe(name):
    def deco(fn):
        def wrapper(*a, **k):
            try:
                fn(*a, **k)
                RESULTS.append((name, "OK", ""))
                print(f"[OK]   {name}")
            except Exception as exc:
                tb = traceback.format_exc(limit=3)
                RESULTS.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))
                print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        return wrapper
    return deco


@probe("very_long_unicode_string")
def t_unicode(td):
    s = ("héllo-世界-" * 10_000) + "🎉"
    p = os.path.join(td, "u.h5")
    hickle.dump({"s": s}, p, mode="w")
    back = hickle.load(p)
    assert back["s"] == s, "unicode round-trip mismatch"


@probe("zero_dim_ndarray")
def t_zero_dim(td):
    a = np.array(3.14, dtype=np.float64)   # 0-d
    p = os.path.join(td, "z.h5")
    hickle.dump({"a": a}, p, mode="w")
    back = hickle.load(p)
    assert back["a"].shape == (), f"shape {back['a'].shape}"
    assert float(back["a"]) == 3.14


@probe("empty_dict_and_list")
def t_empty(td):
    payload = {"empty_dict": {}, "empty_list": [], "empty_tuple": ()}
    p = os.path.join(td, "e.h5")
    hickle.dump(payload, p, mode="w")
    back = hickle.load(p)
    assert back["empty_dict"] == {}, back["empty_dict"]
    assert back["empty_list"] == [], back["empty_list"]
    assert back["empty_tuple"] == (), back["empty_tuple"]


@probe("circular_reference_handled")
def t_cycle(td):
    d = {"a": 1}
    d["self"] = d
    p = os.path.join(td, "c.h5")
    try:
        hickle.dump(d, p, mode="w")
    except (RecursionError, RuntimeError, ValueError, TypeError) as exc:
        # Acceptable: refuse to serialize cycles. Many serializers do.
        print(f"     (refused with {type(exc).__name__} — acceptable)")
        return
    # Or it might silently succeed with broken structure; flag it.
    back = hickle.load(p)
    raise AssertionError(
        f"circular dump did not raise; round-trip type={type(back)}"
    )


@probe("non_ascii_path")
def t_non_ascii_path(td):
    sub = os.path.join(td, "日本語_dir")
    os.makedirs(sub, exist_ok=True)
    p = os.path.join(sub, "data.h5")
    hickle.dump({"x": np.arange(5)}, p, mode="w")
    back = hickle.load(p)
    np.testing.assert_array_equal(back["x"], np.arange(5))


@probe("repeated_dump_state_leak")
def t_repeat(td):
    p = os.path.join(td, "r.h5")
    for i in range(20):
        hickle.dump({"i": i, "arr": np.arange(i + 1)}, p, mode="w")
    back = hickle.load(p)
    assert back["i"] == 19
    assert len(back["arr"]) == 20


@probe("numpy2_float_scalar_subclass")
def t_np2_scalar(td):
    # numpy 2.0 promoted np.float64 etc. to "stringless" repr.
    payload = {"x": np.float64(2.5), "y": np.int32(7), "z": np.bool_(True)}
    p = os.path.join(td, "s.h5")
    hickle.dump(payload, p, mode="w")
    back = hickle.load(p)
    assert float(back["x"]) == 2.5
    assert int(back["y"]) == 7
    assert bool(back["z"]) is True


def main():
    print(f"hickle {hickle.__version__}, numpy {np.__version__}")
    with tempfile.TemporaryDirectory() as td:
        t_unicode(td)
        t_zero_dim(td)
        t_empty(td)
        t_cycle(td)
        t_non_ascii_path(td)
        t_repeat(td)
        t_np2_scalar(td)

    print("\n=== bug-hunt summary ===")
    for name, status, note in RESULTS:
        print(f"  {status:5s}  {name:35s}  {note}")
    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print(f"  failures: {n_fail}/{len(RESULTS)}")
    # Step 7 is informational; do not fail process.
    return 0


if __name__ == "__main__":
    sys.exit(main())
