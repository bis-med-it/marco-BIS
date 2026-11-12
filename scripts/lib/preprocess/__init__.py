"""lib.preprocess — complex preprocessing utilities.

Stateless helpers for operations that don't belong in ``lib.load_dataset``
(which is structural) or ``lib.transform`` (which is per-column univariate).
Cross-series operations, frequency conversions and missing-value
imputation land here.

Implemented:

| Function          | Purpose                       |
|-------------------|-------------------------------|
| ``deflate``       | Nominal → real (divide by price deflator) |
| ``nominalise``    | Real → nominal (multiply by price deflator) |
| ``aggregate``     | Frequency aggregation (e.g., M → Q with per-column flow/rate/stock methods) |
| ``impute``        | Fill missing (ffill / linear / Kalman via VAR(p) state-space; optional EM iteration for heavy missingness via ``em_iter`` kwarg) |

Not included (disaggregation, seasonal adjustment, outlier handling,
standardisation): see ``scripts/lib/preprocess/README.md``.

Scope boundary:

- Per-column univariate transforms (log, diff, log_diff, ...) live in
  ``lib.transform`` — they're a per-model decision applied after loading.
- Trend-cycle filters (HP, Hamilton, BN, KMW-BN) live in ``lib.filters``.
- ``preprocess`` is for operations that span multiple columns, have
  non-trivial state, or wrap heavy third-party tooling. Reusable across
  many models, not tied to one transform decision.

Design principle: stateless functions that take data in, return data out.
``df.attrs`` from ``lib.load_dataset`` should be preserved by wrapping
functions; the underlying primitives (this module) operate on Series.
"""

from .aggregation import aggregate
from .deflation import deflate, nominalise
from .imputation import impute

__all__ = ["aggregate", "deflate", "nominalise", "impute"]
