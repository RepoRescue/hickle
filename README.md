# hickle (RepoRescue modernized fork)

> HDF5-based drop-in replacement for `pickle`. Serialize NumPy arrays,
> pandas frames, nested dicts/lists/tuples to a real `.h5` file and load
> them back bit-exact — across languages, with on-disk compression.

This fork is the upstream
[`telegraphic/hickle`](https://github.com/telegraphic/hickle)
project rescued by the
[**RepoRescue**](https://github.com/RepoRescue) benchmark pipeline so
that it installs and runs cleanly under modern scientific-Python stacks:

- Python **3.13**
- `h5py` **3.16**
- `numpy` **2.x**

The original 5.0.x release on PyPI was last touched before NumPy 2.0
landed and depends on legacy `pkg_resources` import paths. On a fresh
3.13 environment, `hickle.dump` of any NumPy payload now raises
`ValueError: Unable to avoid copy ...` from inside its own loaders.
This fork fixes that without changing the public API.

---

## Install

```bash
pip install -e git+https://github.com/RepoRescue/hickle.git#egg=hickle
```

Or clone and install locally:

```bash
git clone https://github.com/RepoRescue/hickle.git
cd hickle
pip install -e .
```

Requires `python>=3.13`, `h5py>=3.16`, `numpy>=2.0`.

## Quick start

```python
import hickle, numpy as np, tempfile, os

ckpt = {
    "epoch": 17,
    "weights": {
        "encoder.embedding": np.random.randn(128, 64).astype(np.float32),
        "decoder.head.W":    np.random.randn(64, 32).astype(np.float64),
    },
    "metadata": {
        "loss_history": [3.21, 2.05, 1.44, 0.97, 0.61],
        "tags": ["nlp", "transformer"],
        "notes": "Trained on MacBook ☕",
    },
}

with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "checkpoint.h5")
    hickle.dump(ckpt, path, mode="w")          # -> real HDF5 file
    restored = hickle.load(path)
    assert np.array_equal(restored["weights"]["encoder.embedding"],
                          ckpt["weights"]["encoder.embedding"])
    assert restored["metadata"]["notes"] == ckpt["metadata"]["notes"]
```

The result is a genuine HDF5 hierarchy, inspectable with `h5py`, `h5dump`,
`HDFView`, MATLAB, Julia, R — not a re-pickled blob.

## What was actually fixed

Two breakages, surgical patches, no API surface change:

| File | Before | After |
|---|---|---|
| `hickle/loaders/load_numpy.py:235` | `np.array(content, copy=False, dtype=...)` | `np.asarray(content, dtype=...)` |
| `hickle/loaders/load_builtins.py:388` | `np.array(content, copy=False, dtype=...)` | `np.asarray(content, dtype=...)` |
| `hickle/legacy_v3/hickle.py:28` | `from pkg_resources import get_distribution, DistributionNotFound` | `from importlib.metadata import version, PackageNotFoundError` |
| `setup.py` | classifiers / `python_requires` pre-3.13 | bumped for 3.13 + numpy 2 |

The two `np.array(..., copy=False)` call sites are the load hot path:
NumPy 2.0 made `copy=False` raise `ValueError` whenever a copy might be
needed (the old behavior — silently copying — is gone). This breaks every
single hickle load involving a string or non-contiguous buffer. The
patch switches to `np.asarray(..., dtype=...)` which preserves the
"copy only if dtype mismatches" semantics that hickle assumed.

The `pkg_resources -> importlib.metadata` change unblocks `import
hickle.legacy_v3` on 3.13, where `pkg_resources` is deprecated and
absent from many minimal installs.

## Validation

We do not just rerun the upstream test suite. The fork ships two
extra validators under
[`.reporescue/`](.reporescue/) that exercise hickle the way real users do:

### `usability_validate.py` — full round-trip

Builds a heterogeneous nested payload (4 NumPy dtypes, nested
list/tuple, unicode-with-emoji, bool, int), dumps to `.h5`, walks the
file with raw `h5py` to confirm a real HDF5 hierarchy (29 datasets),
loads back, asserts deep structural + bitwise equality.
Engages `hickle.hickle`, `hickle.loaders.load_numpy`,
`hickle.loaders.load_builtins`, `hickle.helpers`, `hickle.lookup`.
Result: PASS.

### `scenario_validate.py` — ML checkpoint resume

A 135-line training scenario:

1. Train a tiny NumPy MLP on a synthetic regression task for 5 epochs.
2. `hickle.dump` `{epoch, weights, metadata}` after each epoch
   (overwrite-latest pattern).
3. Simulate a fresh process: `hickle.load` the snapshot from disk.
4. Resume training for 5 more epochs from the loaded weights.
5. Assert the loss curve keeps decreasing across the resume boundary
   and that the final on-disk weights are bit-equal to in-memory state.

Uses only the public README API (`hickle.dump` / `hickle.load`).
Result: PASS.

### Downstream

The PyPI `hickle` is imported by ~600 public Python files on GitHub.
Star-≥100 active downstreams we identified:

- [`vladfi1/phillip`](https://github.com/vladfi1/phillip) — 583★, last
  push 2025-01-04. Uses hickle in `scripts/merge_data.py` to
  consolidate Smash Bros AI experience replays.
- [`kimbring2/AlphaStar_Implementation`](https://github.com/kimbring2/AlphaStar_Implementation) —
  207★, last push 2024-01.

We did not run these end-to-end (both depend on a
Dolphin emulator + RL stack outside our reach), but the hickle code
paths they exercise — large NumPy ndarray dumps + nested dict
metadata — are exactly what `scenario_validate.py` covers.

## Known limitations (pre-existing, not introduced by the rescue)

We probed seven edge cases (`.reporescue/bug_hunt.py`):

| Probe | Result |
|---|---|
| 100k-char unicode-with-emoji string | OK |
| 0-dim ndarray (`np.array(3.14)`) | OK |
| empty dict / list / tuple | OK |
| **Circular reference** (`d["self"] = d`) | **`dump` silently writes a corrupt file; `load` blows up with `RecursionError`** |
| Non-ASCII path (`日本語_dir/data.h5`) | OK |
| 20× overwrite of same file | OK |
| NumPy 2.x scalar subclasses (`float64`/`int32`/`bool_`) | OK |

The cycle case is a long-standing upstream behavior (no cycle detection
in `dump`), unrelated to this rescue. Realistic checkpoint payloads
are acyclic, so this does not affect the use cases above.

There is also a latent gap in `hickle/legacy_v3/helpers.py`: it still
imports `six`, which is not declared in `requirements.txt`. The
top-level `import hickle` does not touch `legacy_v3`, so the public
API works fine on a clean install. Users who explicitly need to read
hickle-3.x files should `pip install six` themselves until upstream
patches that module too.

## Disclaimer

This is an automated rescue fork produced by the RepoRescue benchmark.
The patch is a minimal-diff modernization for Python 3.13 + NumPy 2.x +
`h5py` 3.16 — it is **not** a feature release and does not vouch for
any behavior beyond what `usability_validate.py` and
`scenario_validate.py` exercise. For the canonical, maintained library,
see [telegraphic/hickle](https://github.com/telegraphic/hickle).

## License

MIT, inherited from the upstream project. See [`LICENSE`](LICENSE).

## Citing hickle

If you use hickle in academic work, please cite the original JOSS paper:

> Price et al. (2018). *Hickle: A HDF5-based python pickle replacement.*
> Journal of Open Source Software, 3(32), 1115.
> https://doi.org/10.21105/joss.01115
