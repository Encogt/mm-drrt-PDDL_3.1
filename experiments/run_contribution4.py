#!/usr/bin/env python
"""Experiment sweep for Contribution 4 (asynchronous refinement and replanning): reuses
demos/demo4.py's own sweep() unchanged, just with defaults sized for a full experimental run
instead of a quick interactive check -- 10 seeds (0-9) instead of 3, and a 10-minute per-(N, seed)
timeout instead of 5 (each run normally takes 5-97s; the larger cap leaves headroom for N=5,6 and
a slower machine). See demos/demo4.py's module docstring for what is being measured; sweep()
already prints mean [min, max] tables with no walkthrough narration.

Usage:
    python experiments/run_contribution4.py                  # N = 2..6, seeds 0-9
    python experiments/run_contribution4.py --sweep 2,3,4    # choose which N values
    python experiments/run_contribution4.py --seeds 0,1,2    # choose which seeds
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'demos')))
from demo4 import sweep  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sweep', type=str, default='2,3,4,5,6', help='Comma-separated N values')
    parser.add_argument('--seeds', type=str, default=','.join(str(s) for s in range(10)),
                        help='Comma-separated seeds (default: 10 seeds, 0-9)')
    parser.add_argument('--run_timeout', type=int, default=600, help='Wall-clock cap (s) per (N, seed) run')
    parser.add_argument('--drrt_time_limit', type=int, default=60,
                        help='dRRT* time limit (s) per refinement attempt')
    parser.add_argument('--refine_attempts', type=int, default=2,
                        help='Refinement attempts (new random seed each) before giving up on this N')
    parser.add_argument('--path_attempts', type=int, default=3,
                        help='Refined paths tried (fresh seed each) until one gets a verified asynchronous '
                             'minimal schedule; the last one is kept if none does')
    args = parser.parse_args()
    sweep(args, [int(x) for x in args.sweep.split(',')], [int(x) for x in args.seeds.split(',')])


if __name__ == '__main__':
    main()
