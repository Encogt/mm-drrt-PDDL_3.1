"""Small JSON-backed cache for region entry/exit offsets, so a repair_region_offsets() widening
from one run has somewhere to land for the NEXT run's measure_region_offsets() to start from --
without this, each run re-measures fresh and forgets any past correction (a violated/widened
result had nowhere to persist to).

Keyed by environment class name, holding {'transit': [alpha, beta], 'transfer': [alpha, beta]} --
deliberately the same coarse, collapsed-per-action-type shape measure_region_offsets() and
repair_region_offsets() already use (not per-robot/per-region -- see
examples/envs/duration_conflict_rai_env.py's measure_region_offsets() docstring for why). One file
for the whole repo; entries namespaced by env class name so different environments don't clobber
each other's cached values.

With main_rai.py's --durations_from_motion, values are [alpha, beta, duration] in seconds of
velocity-limited motion (mm_drrt/utils/motion_timing.py) and stored under '<EnvClass>:motion_time'
-- a different unit system from the index-fraction [alpha, beta] entries, so never mixed with them.
"""
import json
import os

DEFAULT_CACHE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'experiments', 'region_offsets_cache.json'))


def _load_all(cache_path=DEFAULT_CACHE_PATH):
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, 'r') as f:
        return json.load(f)


def load_cached_offsets(env_class_name, cache_path=DEFAULT_CACHE_PATH):
    """{'transit': (alpha, beta), 'transfer': (alpha, beta)} for this environment class (only the
    action types actually cached so far are present), or None if nothing's cached yet at all."""
    entry = _load_all(cache_path).get(env_class_name)
    if entry is None:
        return None
    return {action_type: tuple(value) for action_type, value in entry.items()}


def save_cached_offsets(env_class_name, offsets, cache_path=DEFAULT_CACHE_PATH):
    """Writes offsets (the same {'transit': (alpha, beta), 'transfer': (alpha, beta)} shape
    measure_region_offsets()/repair_region_offsets() already use -- a None value for an action
    type is skipped, not written) for this environment class, merging into whatever's already
    cached for OTHER environment classes rather than clobbering the whole file."""
    all_entries = _load_all(cache_path)
    existing = all_entries.get(env_class_name, {})
    for action_type, value in offsets.items():
        if value is not None:
            existing[action_type] = list(value)
    all_entries[env_class_name] = existing
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(all_entries, f, indent=2)
    return cache_path
