"""Step 5 (partial): the temporal repair half of docs/minimal_temporal_constraints.md's "Future
work" section -- comparing what a region's mutex window was PLANNED with (alpha/beta fed to Tamer
before motion refinement, via sample_region_offsets()) against what the real, plan-specific
dRRT*-refined trajectory ACTUALLY measured (via trajectory_region_fractions() against the executed
path, not the representative sample used at planning time).

Fixed absolute tolerance (not relative to the sub-interval's own width): simpler, and avoids a
tight sub-interval's repair sensitivity scaling down to noise level. See
repair_region_offsets()'s docstring for what counts as a violation under it.

This module does NOT re-solve anything -- a violation is reported as a signal for the caller to
log/flag for re-verification, matching the "widen defensively, don't re-solve" scope agreed on.
"""

from collections import namedtuple

DEFAULT_TOLERANCE = 0.5

RepairResult = namedtuple('RepairResult', ['updated_alpha', 'updated_beta', 'violated',
                                            'entry_diff', 'exit_diff'])


def repair_region_offsets(planned_alpha, planned_beta, measured_alpha, measured_beta,
                          tolerance=DEFAULT_TOLERANCE):
    """Reconciles a region's PLANNED (alpha, beta) -- what Tamer's mutex window was scheduled
    around -- against the MEASURED (alpha, beta) a real, plan-specific dRRT*-refined trajectory
    actually produced for that same action instance.

    Args:
        planned_alpha, planned_beta: the offsets used when this plan was solved (0 <= planned_alpha
            <= planned_beta).
        measured_alpha, measured_beta: the offsets trajectory_region_fractions() measured against
            the REAL executed path for this specific action instance (0 <= measured_alpha <=
            measured_beta).
        tolerance: fixed, absolute seconds. A measured offset within `tolerance` of the planned
            one is treated as agreeing with it (floating-point/sampling noise, not a real
            disagreement) -- not scaled to the sub-interval's own width.

    Returns:
        RepairResult:
          updated_alpha/updated_beta: the offsets to STORE going forward -- the union (min entry,
              max exit) of planned and measured, so a future plan's mutex window only ever widens
              from a real-world observation, never narrows. Always computed, regardless of
              `violated` -- this is the "always safe to do unconditionally" widening.
          violated: True if the actual trajectory's occupancy fell outside
              [planned_alpha, planned_beta] by MORE than `tolerance` on either side -- i.e. the
              mutex window Tamer scheduled THIS run around may not have actually protected the
              real geometry. A violation is a correctness signal about the run that already
              happened (not merely an opportunity to tighten next time); this function only
              reports it, it does not re-solve or re-verify anything itself.
          entry_diff: measured_alpha - planned_alpha (negative means the action entered the region
              EARLIER than planned -- the dangerous direction for alpha).
          exit_diff: measured_beta - planned_beta (positive means the action exited the region
              LATER than planned -- the dangerous direction for beta).

    Raises:
        ValueError: if either (alpha, beta) pair is invalid (alpha > beta).
    """
    if planned_alpha > planned_beta:
        raise ValueError(f"invalid planned interval: planned_alpha={planned_alpha} > "
                         f"planned_beta={planned_beta}")
    if measured_alpha > measured_beta:
        raise ValueError(f"invalid measured interval: measured_alpha={measured_alpha} > "
                         f"measured_beta={measured_beta}")

    entry_diff = measured_alpha - planned_alpha
    exit_diff = measured_beta - planned_beta

    # Dangerous directions only: entering earlier (entry_diff < 0) or exiting later
    # (exit_diff > 0) than planned is what could have let another robot's action overlap the real
    # occupancy window. Entering later / exiting earlier than planned is conservative -- the real
    # occupancy was INSIDE the protected window, never a violation regardless of tolerance.
    violated = (entry_diff < -tolerance) or (exit_diff > tolerance)

    updated_alpha = min(planned_alpha, measured_alpha)
    updated_beta = max(planned_beta, measured_beta)

    return RepairResult(updated_alpha=updated_alpha, updated_beta=updated_beta,
                        violated=violated, entry_diff=entry_diff, exit_diff=exit_diff)


if __name__ == '__main__':
    # Self-check, matching mm_drrt/utils/minimal_temporal_constraint.py's convention (no pytest in
    # this environment -- see that module's self-check block for why this is the pattern here).

    # Measured matches planned exactly: no violation, widened bounds equal the inputs (union of
    # identical pairs is just that pair).
    r = repair_region_offsets(planned_alpha=3.0, planned_beta=7.0, measured_alpha=3.0, measured_beta=7.0)
    assert not r.violated and r.updated_alpha == 3.0 and r.updated_beta == 7.0, r
    assert r.entry_diff == 0.0 and r.exit_diff == 0.0, r

    # Measured within tolerance on both sides: no violation, but the union still widens to the
    # (slightly) larger measured window -- widening is unconditional, violation is not.
    r = repair_region_offsets(planned_alpha=3.0, planned_beta=7.0, measured_alpha=2.8, measured_beta=7.2,
                              tolerance=0.5)
    assert not r.violated, r
    assert r.updated_alpha == 2.8 and r.updated_beta == 7.2, r

    # Measured entered the region EARLIER than planned, beyond tolerance: a real violation (the
    # planned mutex window started too late to have protected this actual trajectory).
    r = repair_region_offsets(planned_alpha=3.0, planned_beta=7.0, measured_alpha=1.0, measured_beta=7.0,
                              tolerance=0.5)
    assert r.violated and r.entry_diff == -2.0, r
    assert r.updated_alpha == 1.0, r  # widened to cover the real, earlier entry

    # Measured exited LATER than planned, beyond tolerance: also a real violation (the planned
    # mutex window ended too early).
    r = repair_region_offsets(planned_alpha=3.0, planned_beta=7.0, measured_alpha=3.0, measured_beta=9.0,
                              tolerance=0.5)
    assert r.violated and r.exit_diff == 2.0, r
    assert r.updated_beta == 9.0, r

    # Measured entered LATER / exited EARLIER than planned (occupancy inside the protected
    # window): conservative, never a violation regardless of how large the gap is.
    r = repair_region_offsets(planned_alpha=3.0, planned_beta=7.0, measured_alpha=4.0, measured_beta=6.0)
    assert not r.violated, r
    assert r.updated_alpha == 3.0 and r.updated_beta == 7.0, r  # union doesn't shrink either

    # Invalid intervals are rejected rather than silently misbehaving.
    for kwargs in [dict(planned_alpha=5.0, planned_beta=2.0, measured_alpha=0.0, measured_beta=10.0),
                  dict(planned_alpha=0.0, planned_beta=10.0, measured_alpha=5.0, measured_beta=2.0)]:
        try:
            repair_region_offsets(**kwargs)
            raise AssertionError(f"expected ValueError for {kwargs}")
        except ValueError:
            pass

    print("All repair_region_offsets() self-checks passed.")
