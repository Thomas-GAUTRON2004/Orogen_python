from types import SimpleNamespace

import pytest

from orogen.v2.tectonics.ending import ENDINGS, frames_ending, history_ending, in_stage
from orogen.v2.tectonics.world import EventKind


CFG = {
    "superShare": 0.8,
    "endingQuietMyr": 20.0,
    "cycleHoldMyr": 50.0,
    "endingBreakupMyr": [10.0, 80.0],
    "endingDispersedShare": 0.5,
    "endingExtraMyr": 100.0,
}


def state(time=100.0, cycle=None, events=None):
    return SimpleNamespace(
        time=time,
        cycle=cycle,
        events=[] if events is None else events,
    )


def test_endings_catalogue_and_none_stage():
    assert [item["id"] for item in ENDINGS] == [
        "none", "supercontinent", "breakup", "dispersed"
    ]
    assert in_stage(state(), "none", CFG)
    assert history_ending(500.0, "none", CFG) == {"untilMyr": 500.0, "stopAt": None}


def test_supercontinent_stage_requires_assembled_share_and_quiet_period():
    cycle = {"state": "assembled", "share": 0.9}
    assert in_stage(state(cycle=cycle), "supercontinent", CFG)
    recent_rift = [{"kind": EventKind["RIFT"], "time": 90.0}]
    assert not in_stage(state(cycle=cycle, events=recent_rift), "supercontinent", CFG)
    old_rift = [{"kind": EventKind["RIFT"], "time": 70.0}]
    assert in_stage(state(cycle=cycle, events=old_rift), "supercontinent", CFG)
    assert not in_stage(state(cycle={"state": "assembled", "share": 0.7}), "supercontinent", CFG)


def test_breakup_stage_uses_first_rift_since_supercontinent_formation():
    events = [
        {"kind": EventKind["SUPERCONTINENT"], "time": 80.0},
        {"kind": EventKind["RIFT"], "time": 40.0},
        {"kind": EventKind["RIFT"], "time": 95.0},
    ]
    assert in_stage(state(events=events), "breakup", CFG)
    assert not in_stage(state(time=200.0, events=events), "breakup", CFG)
    assert not in_stage(state(events=[]), "breakup", CFG)


def test_dispersed_stage_and_unknown_stage():
    assert in_stage(
        state(cycle={"state": "dispersed", "share": 0.4}), "dispersed", CFG
    )
    assert not in_stage(
        state(cycle={"state": "assembled", "share": 0.4}), "dispersed", CFG
    )
    with pytest.raises(ValueError, match="unknown ending"):
        in_stage(state(cycle={}), "invalid", CFG)


def test_history_ending_returns_bounded_stop_callback():
    result = history_ending(100.0, "dispersed", CFG)
    assert result["untilMyr"] == 200.0
    assert not result["stopAt"](state(time=99.0, cycle={"state": "dispersed", "share": 0.4}))
    assert result["stopAt"](state(time=100.0, cycle={"state": "dispersed", "share": 0.4}))


def test_frames_ending_trims_at_first_matching_frame_and_handles_incomplete_runs():
    frames = [
        state(time=0.0),
        state(time=50.0, cycle={"state": "assembled", "share": 0.9}),
        state(time=100.0, cycle={"state": "dispersed", "share": 0.4}),
        state(time=150.0, cycle={"state": "dispersed", "share": 0.4}),
    ]
    assert frames_ending(frames, 75.0, "dispersed", CFG) == frames[:3]
    assert frames_ending(frames[:2], 75.0, "dispersed", CFG) is None
    assert frames_ending(frames[:2], 75.0, "dispersed", CFG, complete=True) == frames[:2]


def test_frames_ending_none_stage_obeys_minimum_time():
    frames = [state(time=0.0), state(time=50.0), state(time=100.0), state(time=150.0)]
    assert frames_ending(frames, 75.0, None, CFG) == frames[:2]
    assert frames_ending(frames[:2], 100.0, None, CFG) is None
