# hickle — Usability Validation

**Selected rescue**: sonnet (srconly: FAIL for sonnet/kimi/glm — full rescue PASS for all three; sonnet preferred per skill priority)
**Scenario type**: E (data parser/pipeline)
**Real-world use**: Drop-in `pickle` replacement that serializes Python/NumPy/pandas/dict/list payloads to a real on-disk **HDF5** file (`.h5`) and round-trips them back. Used in ML to snapshot model checkpoints (weights + metadata) and in scientific computing to dump nested numpy structures portably.

## Step 0: Import sanity
```
repos/rescue_sonnet/hickle/venv-t2/bin/python -c "import hickle"
→ OK (hickle 5.0.3 from rescue tree)
```

## Step 1: Model selection
| model | T2 | T2 srconly |
|---|---|---|
| sonnet | PASS | FAIL |
| kimi   | PASS | FAIL |
| glm    | PASS | FAIL |

All three fail srconly, meaning the rescue depended on dependency pins (`requirements.txt` / `requirements_test.txt`) being applied as well as source changes. Both pieces are present in the rescue archive, so the full-rescue tree is what we validate. Picked **sonnet** by priority.

## Step 2: Scenario rationale
hickle is not a CLI, framework, or analyzer; it is a **data parser/pipeline** (Type E). Its README pitches `hickle.dump(obj, "file.h5") / hickle.load("file.h5")` as drop-in pickle. The validation builds a complex nested object (dict containing numpy arrays of multiple dtypes, nested lists, tuples, bool, unicode strings) and asserts deep structural+bitwise equality on round-trip plus inspects the HDF5 file with `h5py` to confirm it is genuinely an HDF5 hierarchy (29 datasets), not just a re-pickled blob.

## Step 4: Install + core feature (clean venv)
```
python3.13 -m venv /tmp/hickle-clean
/tmp/hickle-clean/bin/pip install -e <rescue tree>          # OK
cd /tmp/hickle-clean                                          # left rescue tree
/tmp/hickle-clean/bin/python <artifacts>/usability_validate.py
```
- `pip install -e` → OK (h5py 3.16.0, numpy 2.4.3, hickle 5.0.3 installed)
- Core feature: `hickle.dump → real HDF5 → hickle.load → deep-equal round-trip`
- Result: **PASS** (final line `USABLE`, see `run.log`)

## Hard constraint 6: Py3.13 / latest-deps surface stressed
| Surface | Evidence |
|---|---|
| `pkg_resources` (deprecated, slated for removal) → `importlib.metadata` | `outputs/sonnet/hickle/hickle.src.patch`: `from pkg_resources import get_distribution, DistributionNotFound` → `from importlib.metadata import version, PackageNotFoundError` in `hickle/legacy_v3/hickle.py:28`. validate.py reads installed `legacy_v3/hickle.py` and asserts the new import is present and `from pkg_resources` is absent. |
| **NumPy 2.0** breaking change: `np.array(..., copy=False)` raises | Patch in `hickle/loaders/load_numpy.py:235` and `hickle/loaders/load_builtins.py:388` rewrites `np.array(content, copy=False, dtype=...)` → `np.asarray(content, dtype=...)`. validate.py exercises the patched line by dumping/reloading `np.array("hellö-world", dtype="U")`, which traverses `load_numpy.create_np_dtype_dataset` → `np.asarray`. Unpatched code raises `ValueError: Unable to avoid copy ...` on numpy 2.x. |
| h5py 3.x string-as-bytes default — would corrupt `"Trained ... ☕"` if not handled | validate.py asserts the unicode string round-trips bit-exact. |
| Updated setup.py classifiers / `python_requires>=3.13` | Patch updates `setup.py` so `pip install -e` resolves on 3.13. |

→ Surface is real; this is **not** a TRIVIAL_RESCUE.

## Hard constraint 5: ≥3 distinct submodules engaged
After validate.py runs, `sys.modules` contains all of:
- `hickle.hickle` (top-level dump/load orchestrator)
- `hickle.loaders.load_numpy` (numpy ndarray dispatch — patched module)
- `hickle.loaders.load_builtins` (dict/list/str dispatch — patched module)
- `hickle.helpers` (PyContainer machinery)
- `hickle.lookup` (type → loader registry)

All asserted explicitly in the script.

## Beyond unit tests (constraint 3)
`hickle/tests/` consists of low-level per-module tests (`test_03_load_builtins.py`, `test_04_load_numpy.py`, etc.) plus `test_hickle.py` — none dumps a structure of the shape used here (heterogeneous nested dict with 4 dtypes of ndarray + unicode emoji + nested list + tuple + bool, then re-opens the HDF5 file with raw h5py to enumerate datasets). Validate.py also reads installed `legacy_v3/hickle.py` source to verify the rescue patch text — no test does that.

```
$ grep -rn "visititems" repos/rescue_sonnet/hickle/hickle/tests/
(no matches)
```

## Step 6: Downstream / Scenario
- **Path A** (downstream): GitHub code search `"import hickle" language:Python` → 600 hits. Star-≥100 active downstreams found:
  - `vladfi1/phillip` — 583★, last push 2025-01-04 (within 2-year window). Uses hickle in `scripts/merge_data.py` to consolidate Smash Bros AI experience replays.
  - `msracver/FCIS` — 1565★, last push 2021 (outside 2y).
  - `kimbring2/AlphaStar_Implementation` — 207★, last push 2024-01.

  Cannot run phillip end-to-end (depends on a Dolphin emulator + RL stack), so the workflow itself is documented but not executed. Path A is identified-but-not-fully-runnable.
- **Path B** (scenario, ≥30 lines): `scenario_validate.py` (~135 lines) implements a full ML checkpoint loop — train tiny NumPy MLP for 5 epochs, hickle-dump a `{epoch, weights, metadata}` ckpt each step, simulate a fresh process, `hickle.load` the latest ckpt, resume for 5 more epochs, assert loss curve monotonically decreases across the resume boundary and final hickle-loaded weights are bit-equal to in-memory final state. Runs **PASS** in clean venv outside rescue tree (see `run.log`). Only public README API used.

→ Constraint 8 satisfied via Path B; Path A documented.

## Step 7: Bug-hunt
Tried 7 edge cases (`bug_hunt.py`):

| probe | result |
|---|---|
| very long unicode + emoji string (~100k chars) | OK |
| 0-dim ndarray (`np.array(3.14)`) | OK |
| empty dict / list / tuple | OK |
| **circular reference** (`d["self"]=d`) | **dump silently succeeds, load explodes with `RecursionError → ValueError`** |
| non-ASCII path (`日本語_dir/data.h5`) | OK |
| 20× repeated overwrite of same file (state leak) | OK |
| numpy 2.x scalar subclasses (`np.float64`, `np.int32`, `np.bool_`) | OK |

The one finding: hickle `dump` does not detect cycles; the `.h5` is written but is corrupt and only blows up on `load`. This is a **pre-existing** library issue, not introduced by the rescue, and does not affect the USABLE verdict — every realistic checkpoint use case is acyclic.

Side observation: `import hickle.legacy_v3.hickle` fails on 3.13 because `legacy_v3/helpers.py` still imports `six`, which is not in `requirements.txt`. The rescue did patch `legacy_v3/hickle.py` (pkg_resources → importlib.metadata) but missed the `six` dependency. Top-level `import hickle` does not touch `legacy_v3` so the public API is unaffected; this is a latent issue only triggered by users explicitly loading legacy v3 hickle files.

## Verdict
STATUS: USABLE

Reason: Clean-venv `pip install -e` succeeds, the README primary use mode (`hickle.dump → real HDF5 → hickle.load`) round-trips a heterogeneous nested payload bit-exact, ≥3 submodules are exercised, the patched 3.13/numpy-2.0 surface is provably traversed (numpy `asarray` path + verified `importlib.metadata` source rewrite), and a 135-line ML-checkpoint scenario runs end-to-end. Bug-hunt found one pre-existing library limitation (circular refs) and one latent legacy_v3 dependency gap (`six`); neither affects the public hickle API.
