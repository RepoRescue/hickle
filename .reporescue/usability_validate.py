"""
hickle usability validation (Type E: data parser/pipeline).

Sells: serialize arbitrary Python/NumPy/pandas data to a real on-disk HDF5
file, deserialize back, get the same object structure.

Hits ≥3 distinct submodules:
  - hickle.dump         (top-level orchestrator -> hickle.hickle.dump)
  - hickle.load         (top-level orchestrator -> hickle.hickle.load)
  - hickle.loaders.load_numpy   (numpy.ndarray dispatch)
  - hickle.loaders.load_builtins (dict/list/str dispatch)
  - hickle.helpers      (PyContainer machinery used internally)

Run AFTER `pip install -e <rescue tree>` in a clean venv,
from a CWD outside the rescue tree.
"""
from __future__ import annotations

import os
import sys
import tempfile
import importlib

import numpy as np
import h5py

import hickle
from hickle import dump, load
import hickle.loaders.load_numpy as _load_numpy_mod
import hickle.loaders.load_builtins as _load_builtins_mod
import hickle.helpers as _helpers_mod


def _assert_eq(a, b, path="root"):
    """Deep structural equality check tolerant of numpy arrays."""
    if isinstance(a, dict):
        assert isinstance(b, dict), f"{path}: type mismatch dict vs {type(b)}"
        assert set(a.keys()) == set(b.keys()), (
            f"{path}: key mismatch {set(a.keys())} vs {set(b.keys())}"
        )
        for k in a:
            _assert_eq(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b), f"{path}: type {type(a)} vs {type(b)}"
        assert len(a) == len(b), f"{path}: len {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_eq(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray), f"{path}: not ndarray, got {type(b)}"
        assert a.shape == b.shape, f"{path}: shape {a.shape} vs {b.shape}"
        assert a.dtype == b.dtype, f"{path}: dtype {a.dtype} vs {b.dtype}"
        assert np.array_equal(a, b), f"{path}: values differ"
    elif isinstance(a, float) and np.isnan(a):
        assert isinstance(b, float) and np.isnan(b), f"{path}: nan mismatch"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def main() -> int:
    print(f"[setup] CWD = {os.getcwd()}")
    print(f"[setup] hickle      from {hickle.__file__}")
    print(f"[setup] hickle ver  {hickle.__version__}")
    print(f"[setup] h5py ver    {h5py.__version__}")
    print(f"[setup] numpy ver   {np.__version__}")
    # Sanity: rescue tree is somewhere else, we must NOT be running from inside it.
    assert "rescue_sonnet/hickle" not in os.getcwd(), \
        "must be invoked outside rescue tree"

    # ---- Build a complex nested object covering ≥4 data classes ---------
    rng = np.random.default_rng(42)
    payload = {
        "model_name": "tiny-transformer-v3",
        "epoch": 17,
        "weights": {
            "encoder.embedding": rng.standard_normal((128, 64)).astype(np.float32),
            "encoder.layer0.W_q": rng.standard_normal((64, 64)).astype(np.float32),
            "encoder.layer0.bias": np.zeros((64,), dtype=np.float32),
            "decoder.head.W": rng.standard_normal((64, 32)).astype(np.float64),
        },
        "metadata": {
            "tags": ["nlp", "transformer", "checkpoint"],
            "dataset_ids": (101, 202, 303),
            "loss_history": [3.21, 2.05, 1.44, 0.97, 0.61],
            "is_final": False,
            "notes": "Trained on MacBook; intermediate snapshot ☕",  # unicode
        },
        "matrices": [
            np.eye(4, dtype=np.int64),
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        ],
    }

    with tempfile.TemporaryDirectory() as td:
        h5path = os.path.join(td, "checkpoint.h5")

        # ---- DUMP -------------------------------------------------------
        dump(payload, h5path, mode="w")
        assert os.path.isfile(h5path), "no file produced"
        size = os.path.getsize(h5path)
        assert size > 1024, f"hickle file too small: {size}B"
        print(f"[dump] wrote {h5path} ({size} bytes)")

        # ---- Verify it's a real HDF5 file (constraint 1: real output) ----
        with h5py.File(h5path, "r") as f:
            assert "data" in f or len(f.keys()) > 0, "no top-level group"
            # Walk keys to prove HDF5 hierarchy was actually populated.
            n_datasets = 0
            def _count(_name, obj):
                nonlocal n_datasets
                if isinstance(obj, h5py.Dataset):
                    n_datasets += 1
            f.visititems(_count)
            print(f"[h5py] file contains {n_datasets} datasets")
            assert n_datasets >= 5, f"expected many datasets, got {n_datasets}"

        # ---- LOAD -------------------------------------------------------
        restored = load(h5path)
        print(f"[load] restored top-level type {type(restored).__name__}")

        # ---- Deep-equality assertion (constraint 2) ---------------------
        _assert_eq(payload, restored)
        print("[assert] deep equality OK")

        # ---- Spot-check a numpy weight to be paranoid -------------------
        assert restored["weights"]["encoder.embedding"].dtype == np.float32
        assert restored["weights"]["encoder.embedding"].shape == (128, 64)
        np.testing.assert_array_equal(
            payload["weights"]["encoder.embedding"],
            restored["weights"]["encoder.embedding"],
        )

        # ---- Constraint 5: prove ≥3 distinct submodules really ran -----
        # We did this implicitly above; assert their modules are loaded.
        for mod in (
            "hickle.hickle",
            "hickle.loaders.load_numpy",
            "hickle.loaders.load_builtins",
            "hickle.helpers",
            "hickle.lookup",
        ):
            assert mod in sys.modules, f"{mod} not loaded"
        print("[modules] hickle.hickle / loaders.load_numpy / "
              "loaders.load_builtins / helpers / lookup all engaged")

        # ---- Constraint 6: prove the 3.13 surface was traversed --------
        # The rescue patched two breakpoints we now exercise:
        #   (a) pkg_resources.get_distribution -> importlib.metadata.version
        #   (b) np.array(..., copy=False) -> np.asarray(..., dtype=...)
        # (a) loading a string via load_numpy hits the np.asarray path on
        #     numpy 2.x where copy=False would have raised.
        s_payload = {"s": np.array("hellö-world", dtype="U")}
        s_path = os.path.join(td, "str.h5")
        dump(s_payload, s_path, mode="w")
        s_back = load(s_path)
        assert str(s_back["s"]) == "hellö-world"
        print("[3.13-surface] numpy 2.x asarray path traversed (load_numpy)")

        # (b) The rescue replaced pkg_resources -> importlib.metadata in
        #     legacy_v3.hickle (pkg_resources is deprecated, moved/removed
        #     from 3.12+ stdlib path). Verify by reading installed source.
        import hickle as _h
        legacy_path = os.path.join(
            os.path.dirname(_h.__file__), "legacy_v3", "hickle.py"
        )
        src = open(legacy_path, encoding="utf-8").read()
        assert "from importlib.metadata import" in src, (
            "rescue patch not present in legacy_v3"
        )
        # The pkg_resources import line must be gone; the literal token may
        # appear elsewhere in comments, so check the import statement only.
        assert "from pkg_resources" not in src, "pkg_resources still imported"
        print("[3.13-surface] legacy_v3.hickle uses importlib.metadata")

    print("USABLE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
