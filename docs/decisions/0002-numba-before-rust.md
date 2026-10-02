# 0002 – Numba before Rust for compiled rainflow counting

- **Status:** Accepted
- **Date:** 2026-10-02
- **Release:** 5.6 (Fast fatigue)
- **Revisit:** if 6.x needs compiled readers or lazy loading of large files

## Context

Rainflow counting in `qats/fatigue/rainflow.py` (ASTM E1049-85) is pure Python. The `reversals()` generator walks every sample of the series, and `cycles()` runs a stack-based (deque) loop over the reversals. It is correct but slow on long series and on many series. The roadmap's 5.6 target is at least a 10× speed-up on a 3-hour, 10 Hz series with identical results.

The code that needs speed is small: about 200 lines of reversal detection and cycle extraction. The question is how to compile it.

## Options

### Numba

You add a decorator (`@njit`) to a Python function. On first call it compiles that function to machine code with LLVM. The code stays Python.

### Rust (PyO3 + maturin)

You write the loop in Rust and expose it to Python with PyO3. It's built ahead of time into binary wheels with maturin, so QATS stops being a pure-Python package.

### Considered briefly

- **Cython** has the same platform-wheel burden as Rust, with fewer safety guarantees and no other clear advantage here.
- **Pure NumPy vectorisation** isn't possible for the counting itself, because the stack-based algorithm is inherently sequential. It *is* possible for reversal detection (see step 1 under Decision).

## Gains and pains

| | Numba | Rust (PyO3 + maturin) |
|---|---|---|
| **Speed** | 10–100× over pure Python for loops like rainflow | About the same as Numba for this loop; somewhat more predictable |
| **Code** | Stays Python, so both maintainers can read and review it directly | A second language; reviewing correctness needs Rust knowledge, even with AI help |
| **First call** | Compile delay of a second or so; cacheable to disk with `cache=True` | None |
| **Install size** | Pulls in `numba` + `llvmlite`, tens of MB | Adds well under 1 MB |
| **New Python versions** | Support usually lags a new Python release by weeks to months. That clashes with the 5.4 goal "works on the latest Python" | You control it; one abi3 wheel per platform covers all newer Python versions |
| **NumPy versions** | Numba caps the NumPy versions it supports, which can block users from upgrading NumPy | No such coupling |
| **Packaging** | Unchanged: still a pure-Python wheel | Platform wheels for Windows, Linux and macOS (x86 + ARM) built in CI; Rust compilers needed for conda-forge; switch to the maturin build backend |
| **Contributors** | Any Python engineer can contribute | Few engineering users write Rust, which raises the bar for outside contributions |
| **Room to grow** | Fine for numeric loops; awkward for anything else | Could also take on fast file parsing and lazy readers later |

## Reasoning

1. **Numba's biggest pain can be avoided by making it optional.** Use `pip install qats[fast]` to get Numba; without it, the same functions run as plain Python. Users never get blocked by Numba lagging a Python or NumPy release; they just get the slower path for a while.
2. **Rust's pains are mostly one-off packaging work** (wheel builds, a new build backend) plus a lasting cost in contributors and code review. That's worth it only if a lot more of QATS goes native, not for one function.
3. **There's a cheap step before either option.** `qats.signal.find_reversals` already finds reversals with vectorised NumPy. Feeding its output to the counting loop shrinks the work to the reversals only, often 5–10× fewer points than the raw series. This alone may give much of the speed-up at no cost.
4. **Team fit.** QATS has two maintainers who both write Python. Keeping the code in Python keeps it maintainable by both and open to contributions from engineering users.

## Decision

Try Numba first, in three steps:

1. **Benchmark and vectorise reversals.** Add a benchmark (a 3-hour series at 10 Hz, plus a batch of many series) to the test suite. Then replace the per-sample `reversals()` generator with the vectorised `qats.signal.find_reversals` output as input to the counting loop. Measure.
2. **Numba as an optional extra.** Compile the counting loop with `@njit(cache=True)`, declared as the extra `qats[fast]`. If Numba can't be imported, fall back to the pure-Python implementation with identical results. Don't raise a warning on import; mention the extra in the docs.
3. **CI tests both paths.** Run the rainflow tests with and without Numba installed, and assert that results are identical.

## Consequences

- QATS stays a pure-Python wheel. Packaging, conda-forge and the planned Windows installer are unaffected (the installer may bundle Numba).
- Numba may lag new Python releases. In that case `qats[fast]` stays installable on older Python only, and the fallback covers the newest versions.
- Two code paths must be kept in sync. The shared test suite and the identical-results assertion guard this.
- Revisit Rust if 6.x needs fast readers or lazy loading of large files, where a compiled core pays off more widely. A new decision record would replace this one.

## Open points

- How often do users actually run rainflow counting on very long or many series? The benchmark in step 1 should reflect those real workloads.
