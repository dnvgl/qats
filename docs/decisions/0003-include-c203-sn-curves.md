# 0003 – Include DNV-RP-C203 S-N curves in the library

- **Status:** Accepted
- **Date:** 2026-10-02
- **Release:** 5.6 (Fast fatigue)

## Context

`qats.fatigue.sn.SNCurve` already implements the DNV-RP-C203 formulation: bilinear curves, the fatigue limit and the thickness correction (eq. 2.4.3). But users must type in every curve's parameters (`m1`, `m2`, `loga1`, `loga2`, `t_ref`, `k` and so on) themselves. That is tedious and a source of input errors, and it's the most common starting point for fatigue checks in our user group.

The open question was whether we can ship the curve tables with QATS, or whether users must supply curves themselves.

## Decision

Include the DNV-RP-C203 S-N curves as ready-made `SNCurve` presets.

## Implementation notes

- Organise presets by environment, as the standard does: in air, in seawater with cathodic protection, and free corrosion. Look them up by name, for example `sn.get_curve("D", env="air")`; the final API is to be settled in the implementation PR.
- Record the edition of DNV-RP-C203 that the values come from on each preset and in the docs. When a new edition changes values, add the new presets and keep the old ones selectable.
- Test every preset against values in the standard's tables: `loga1`, `loga2`, `m1`, `m2`, fatigue limit and thickness exponent.
- Ship only the curve parameters (numbers), not reproduced text or figures from the standard, and reference the standard in docstrings.

## Consequences

- Fatigue calculations become quicker to set up and less error-prone.
- The presets must be kept in step with new editions of the standard.
- The same pattern can later be used for other curve sets (for example API or BS 7608) if users ask.
