"""Shared helpers for the GUI walkthroughs in demos/: narration that pauses for Enter, scene setup,
region highlighting, ASCII Gantt charts and the two replay modes. Every demo also runs headlessly
with --no_gui (narration auto-advances, no viewer, no replay)."""
import argparse
import os
import random
import sys
import time
import textwrap

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import robotic as ry  # noqa: E402

from mm_drrt.pipeline_rai import build_parser, make_env, release_targets_and_grippers, replay_lockstep  # noqa: E402
from mm_drrt.utils.rai_utils import connect, disconnect, refresh_view  # noqa: E402
from mm_drrt.utils.rai_motion_planner_utils import replay_timed  # noqa: E402
from mm_drrt.utils.coordination_regions import region_volume_from_frame  # noqa: E402

WIDTH = 96


def demo_args(description, extra=None):
    """Common demo flags: --no_gui, --seed, --speed (async replay speed-up)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--no_gui', action='store_true', help='Headless: no viewer, narration auto-advances')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--speed', type=float, default=1.0, help='Replay speed-up for timed (asynchronous) replays')
    if extra:
        extra(parser)
    return parser.parse_args()


class Walkthrough(object):
    def __init__(self, title, use_gui):
        self.use_gui = use_gui
        self.n = 0
        print('=' * WIDTH)
        print(title.center(WIDTH))
        print('=' * WIDTH)

    def step(self, title, text=''):
        """A numbered narration step: prints the explanation, then waits for Enter (GUI mode)."""
        self.n += 1
        print('\n' + '-' * WIDTH)
        print(f"[{self.n}] {title}")
        print('-' * WIDTH)
        if text:
            for para in textwrap.dedent(text).strip().split('\n\n'):
                print(textwrap.fill(' '.join(para.split()), WIDTH))
                print()
        self.pause()

    def pause(self, prompt='  (press Enter to continue)'):
        if self.use_gui:
            input(prompt)

    def say(self, text):
        for para in textwrap.dedent(text).strip().split('\n\n'):
            print(textwrap.fill(' '.join(para.split()), WIDTH))


def pipeline_opt(use_gui, seed, *flags):
    """main_rai.py's option namespace for these flags (--use_gui is store_false there)."""
    argv = list(flags) + ['--seed', str(seed)] + ([] if use_gui else ['--use_gui'])
    return build_parser().parse_args(argv)


def setup_scene(opt):
    """Seeded scene + viewer for one pipeline option set. Returns (C, env)."""
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    C = connect(use_gui=opt.use_gui)
    env = make_env(opt, C)
    refresh_view(C, use_gui=opt.use_gui)
    return C, env


def close_scene(C):
    disconnect(C)


def show_region(C, frame, color=(1.0, 0.2, 0.2), alpha=0.25, use_gui=True):
    """Draws a region's coordination volume (coordination_regions.region_volume_from_frame) as a
    translucent, non-colliding box, so the viewer shows what 'occupying the region' means."""
    region = region_volume_from_frame(C, frame)
    name = f'region_volume_{frame}'
    if name not in C.getFrameNames():
        C.addFrame(name).setPosition(list(region.center)).setShape(
            ry.ST.box, size=[2 * h for h in region.half_extents]).setColor(list(color) + [alpha]).setContact(0)
        if use_gui:
            C.view_recopyMeshes()
            C.view(False)
    return region


def robot_index(env, robot):
    return list(env.robots.values()).index(robot)


def gantt(rows, t_max=None, width=60, title=None):
    """rows: [(label, [(name, start, end, occ_start or None, occ_end or None)])]. Draws each
    action as '=' and the part of it that occupies the shared region as '#'."""
    spans = [x for _, acts in rows for x in acts]
    if not spans:
        return
    t_max = t_max or max(e for _, _, e, _, _ in spans)
    scale = width / t_max if t_max > 0 else 1.0
    if title:
        print(title)
    for label, acts in rows:
        line = [' '] * (width + 1)
        names = []
        for name, s, e, os_, oe in sorted(acts, key=lambda a: a[1]):
            i0, i1 = int(round(s * scale)), max(int(round(e * scale)), int(round(s * scale)) + 1)
            for i in range(i0, min(i1, width + 1)):
                line[i] = '='
            if os_ is not None:
                j0, j1 = int(round(os_ * scale)), max(int(round(oe * scale)), int(round(os_ * scale)) + 1)
                for i in range(j0, min(j1, width + 1)):
                    line[i] = '#'
            names.append(f"{name}[{s:.2f}-{e:.2f}]")
        print(f"  {label:>4} |{''.join(line)}| {' '.join(names)}")
    print(f"  {'':>4}  0{'':{width - 8}}{t_max:7.2f}s    (= action, # region occupancy)")


def schedule_rows(env, schedule, occupancy=None):
    """Gantt rows from a solved schedule (pddl_parser._build_schedule). occupancy: optional
    {name: (alpha, beta)} relative to each action's start; defaults to the schedule's own occupy
    phase when it was split, i.e. when the region mutex narrowed it."""
    rows = {}
    robots = list(env.robots.values())
    for a in schedule:
        # Abstract problems (e.g. compare_region_fluents_demo's) name robots by plain strings.
        r = f"r{robots.index(a['robot'])}" if a['robot'] in robots else str(a['robot'])
        if occupancy is not None:
            occ = occupancy.get(a['name'])
            os_, oe = (a['start'] + occ[0], a['start'] + occ[1]) if occ else (None, None)
        else:
            os_, oe = a['occupy_start'], a['occupy_end']
        rows.setdefault(r, []).append((a['name'], a['start'], a['end'], os_, oe))
    return [(r, rows[r]) for r in sorted(rows)]


def timed_rows(timeline, starts):
    """Gantt rows for an execution where action n starts at starts[n] and runs timeline[n]['duration']."""
    rows = {}
    for n, a in timeline.items():
        s = starts[n]
        occ = (s + a['entry'], s + a['exit']) if a['entry'] is not None else (None, None)
        rows.setdefault(a['robot_index'], []).append((n, s, s + a['duration'], occ[0], occ[1]))
    return [(f"r{r}", rows[r]) for r in sorted(rows)]


def lockstep_rows(timeline):
    """Gantt rows for the synchronous (lock-step) execution: each action spans its absolute
    start..end, waits at composite nodes included, with its region occupancy at the absolute times
    of the waypoints where the gripper enters/leaves."""
    rows = {}
    for n, a in timeline.items():
        occ = (None, None)
        if a['entry'] is not None:
            times = list(a['motion_times'])
            occ = (a['abs_times'][times.index(a['entry'])], a['abs_times'][times.index(a['exit'])])
        rows.setdefault(a['robot_index'], []).append((n, a['start'], a['end'], occ[0], occ[1]))
    return [(f"r{r}", rows[r]) for r in sorted(rows)]


def replay_sync(C, env, plan, composite_path, use_gui):
    if not use_gui:
        return
    start = time.time()
    replay_lockstep(C, env, plan, composite_path)
    print(f"  (lock-step replay took {time.time() - start:.1f}s wall-clock)")


def replay_async(C, env, plan, execution, use_gui, speed=1.0):
    if not use_gui:
        return
    robots = list(env.robots.values())
    release_targets, gripper_frames = release_targets_and_grippers(env, plan)
    replay_timed(C, execution, env.get_joints(robots), release_targets, gripper_frames, speed=speed)


def table(headers, rows):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    print('  ' + '  '.join(str(h).rjust(w) for h, w in zip(headers, widths)))
    print('  ' + '  '.join('-' * w for w in widths))
    for r in rows:
        print('  ' + '  '.join(str(c).rjust(w) for c, w in zip(r, widths)))
