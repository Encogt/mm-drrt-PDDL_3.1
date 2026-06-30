"""
ANML Temporal Domain Definition for MM-dRRT Manipulation Tasks.

Provides the ANML domain as a string so it can be written to a temp file at
planning time with configurable action durations.
"""

import os


def build_mm_drrt_anml_domain(transit_duration=10, transfer_duration=10):
    return f"""type Robot;
type MovableObj;
type FixedObj;

fluent boolean robot_free(Robot r);
fluent boolean holding(Robot r, MovableObj m);
fluent boolean obj_clear(MovableObj m);
fluent boolean surface_accessible(FixedObj f);
fluent boolean robot_can_reach(Robot r, FixedObj f);
fluent FixedObj obj_location(MovableObj m);

action transit(Robot r, MovableObj m, FixedObj from_f) {{
  duration := {transit_duration};
  [start]   robot_free(r);
  [start]   obj_location(m) == from_f;
  [start]   obj_clear(m);
  [all]     surface_accessible(from_f);
  [all]     robot_can_reach(r, from_f);
  [start]   robot_free(r) := false;
  [start]   obj_clear(m)  := false;
  [end]     holding(r, m) := true;
}};

action transfer(Robot r, MovableObj m, FixedObj to_f) {{
  duration := {transfer_duration};
  [start]   holding(r, m);
  [all]     surface_accessible(to_f);
  [all]     robot_can_reach(r, to_f);
  [end]     robot_free(r)   := true;
  [end]     holding(r, m)   := false;
  [end]     obj_clear(m)    := true;
  [end]     obj_location(m) := to_f;
}};
"""


def get_domain_file_path():
    domain_file = os.path.join(
        os.path.dirname(__file__),
        '../anml/domains/mm_drrt_manipulation.anml'
    )
    return os.path.abspath(domain_file)
