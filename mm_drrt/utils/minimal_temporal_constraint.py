"""C3: the minimal temporal constraint derivation.

Given two high-level actions i and j that both occupy the same coordination region k -- i during
[si + alpha_ik, si + beta_ik], j during [sj + alpha_jk, sj + beta_jk], relative to their own
(still-unknown) start times si, sj -- exactly two orderings keep the two occupancy intervals from
overlapping:

    i before j:  si + beta_ik <= sj + alpha_jk   <=>   sj - si >= beta_ik - alpha_jk
    j before i:  sj + beta_jk <= si + alpha_ik   <=>   si - sj >= beta_jk - alpha_ik

Forcing one fixed direction regardless of the actual numbers is exactly the FULL ACTION
SERIALIZATION this whole feature exists to avoid (see mm_drrt/planner/pddl_domain.py's docstring).
derive_minimal_constraint() instead picks whichever ordering requires the SMALLER enforced gap --
the tighter, less restrictive of the two valid constraints -- so the region mutex costs only as
much scheduling freedom as the real geometry actually demands.
"""

from collections import namedtuple

MinimalConstraint = namedtuple('MinimalConstraint', ['order', 'gap', 'raw_gap_i_before_j',
                                                      'raw_gap_j_before_i', 'free'])


def derive_minimal_constraint(alpha_ik, beta_ik, alpha_jk, beta_jk):
    """Derives the minimal (si, sj) separation constraint that keeps action i's [alpha_ik, beta_ik]
    and action j's [alpha_jk, beta_jk] occupancy of the same region from overlapping.

    Args:
        alpha_ik, beta_ik: action i's region-entry/exit offsets (0 <= alpha_ik <= beta_ik)
        alpha_jk, beta_jk: action j's region-entry/exit offsets (0 <= alpha_jk <= beta_jk)

    Returns:
        MinimalConstraint:
          order: 'i_before_j' (enforce si + beta_ik <= sj + alpha_jk) or 'j_before_i' (enforce
              sj + beta_jk <= si + alpha_ik) -- whichever requires the smaller gap. Arbitrary
              (kept as 'i_before_j') when `free` is True, since neither direction needs enforcing.
          gap: the minimum separation (sj - si for i_before_j, si - sj for j_before_i) that must
              actually be enforced for the chosen order -- clamped to 0.0, since a negative raw
              gap only means the two intervals already don't overlap even at si == sj (nothing to
              enforce, not a real requirement to go further negative).
          raw_gap_i_before_j, raw_gap_j_before_i: the two orderings' unclamped required gaps
              (beta_ik - alpha_jk and beta_jk - alpha_ik respectively), for callers that want the
              full picture rather than just the chosen minimal constraint.
          free: True if BOTH raw gaps are <= 0 -- the two actions can start at the exact same time
              (si == sj) and their region occupancy still never overlaps at all, so no ordering
              constraint is needed between them in this region. (In practice this requires
              degenerate/near-zero-width occupancy on at least one side; see this module's tests.)

    Raises:
        ValueError: if alpha_ik > beta_ik or alpha_jk > beta_jk (an invalid occupancy interval).
    """
    if alpha_ik > beta_ik:
        raise ValueError(f"invalid occupancy interval for i: alpha_ik={alpha_ik} > beta_ik={beta_ik}")
    if alpha_jk > beta_jk:
        raise ValueError(f"invalid occupancy interval for j: alpha_jk={alpha_jk} > beta_jk={beta_jk}")

    raw_gap_i_before_j = beta_ik - alpha_jk
    raw_gap_j_before_i = beta_jk - alpha_ik

    if raw_gap_i_before_j <= raw_gap_j_before_i:
        order, raw_gap = 'i_before_j', raw_gap_i_before_j
    else:
        order, raw_gap = 'j_before_i', raw_gap_j_before_i

    return MinimalConstraint(
        order=order,
        gap=max(raw_gap, 0.0),
        raw_gap_i_before_j=raw_gap_i_before_j,
        raw_gap_j_before_i=raw_gap_j_before_i,
        free=raw_gap_i_before_j <= 0 and raw_gap_j_before_i <= 0,
    )


if __name__ == '__main__':
    # Self-check: no pytest in this environment (see mm_drrt/planner/pddl_domain.py's neighbors --
    # none of this repo's other new modules have a pytest-based test file either), so this runs a
    # few concrete cases directly, matching the plain-assert style already used for validation
    # elsewhere in this feature (e.g. compare_region_constraints.py's invariant checks).

    # Symmetric case: equal-width intervals -> either order needs the same gap; picks i_before_j.
    c = derive_minimal_constraint(alpha_ik=3, beta_ik=7, alpha_jk=3, beta_jk=7)
    assert c.order == 'i_before_j' and c.gap == 4.0 and not c.free, c

    # Asymmetric case: i occupies almost the whole action (0-10), j only occupies a late sliver
    # (8-10). Forcing "j before i" would need si >= sj + 10 -- nearly a full extra action's worth
    # of delay. The minimal constraint instead lets i go first for a tiny 2-unit gap.
    c = derive_minimal_constraint(alpha_ik=0, beta_ik=10, alpha_jk=8, beta_jk=10)
    assert c.order == 'i_before_j' and c.gap == 2.0, c
    assert c.raw_gap_j_before_i == 10.0, c  # what "full action serialization" would have cost

    # Reverse of the above: j occupies almost the whole action, i only occupies a late sliver.
    c = derive_minimal_constraint(alpha_ik=8, beta_ik=10, alpha_jk=0, beta_jk=10)
    assert c.order == 'j_before_i' and c.gap == 2.0, c

    # Full-duration mutex (alpha=0, beta=duration for both, matching pddl_domain.py's default
    # collapse): the minimal constraint reduces exactly to "one whole action before the other,"
    # since there IS no sub-interval to exploit here -- this is the expected worst case, not a bug.
    c = derive_minimal_constraint(alpha_ik=0, beta_ik=10, alpha_jk=0, beta_jk=10)
    assert c.gap == 10.0, c

    # i's occupancy is a single instant that happens strictly before j's even begins, at si == sj:
    # the i_before_j direction is already satisfied with zero gap. j_before_i is NOT (it would
    # need a real 5-unit gap) -- only one side is free here, so the algorithm still correctly picks
    # the other, already-satisfied side rather than reporting the whole pair as unconstrained.
    c = derive_minimal_constraint(alpha_ik=0, beta_ik=0, alpha_jk=5, beta_jk=5)
    assert c.order == 'i_before_j' and c.gap == 0.0 and not c.free, c

    # Both sides zero-width AND coincident: `free` is True (see this module's docstring on how
    # degenerate that combination actually is -- both intervals must collapse to the same instant).
    c = derive_minimal_constraint(alpha_ik=5, beta_ik=5, alpha_jk=5, beta_jk=5)
    assert c.free and c.gap == 0.0, c

    # Invalid interval is rejected rather than silently misbehaving.
    try:
        derive_minimal_constraint(alpha_ik=5, beta_ik=2, alpha_jk=0, beta_jk=10)
        raise AssertionError("expected ValueError for alpha_ik > beta_ik")
    except ValueError:
        pass

    print("All derive_minimal_constraint() self-checks passed.")
