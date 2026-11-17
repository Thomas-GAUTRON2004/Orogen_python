"""History-ending predicates and frame selection ported from Orogen V2."""
from __future__ import annotations

from .world import EventKind

ENDINGS = [
    {"id": "none", "label": "Wherever the history has got to", "phrase": "wherever it has got to"},
    {"id": "supercontinent", "label": "A supercontinent (a Pangaea)", "phrase": "a supercontinent"},
    {"id": "breakup", "label": "A supercontinent breaking up (young rift seas)", "phrase": "a supercontinent breaking up"},
    {"id": "dispersed", "label": "Continents dispersed (as on Earth today)", "phrase": "dispersed continents"},
]


def in_stage(state, stage: str, config: dict) -> bool:
    """Return whether a world or snapshot is in the requested tectonic-cycle stage."""
    if stage == "none":
        return True
    cycle = state.cycle
    time, events = float(state.time), state.events
    if cycle is None:
        return False

    def last_of(kind):
        for event in reversed(events):
            if event["kind"] == kind and float(event["time"]) <= time + 1e-9:
                return event
        return None

    if stage == "supercontinent":
        rift = last_of(EventKind["RIFT"])
        return (
            cycle["state"] == "assembled"
            and float(cycle["share"]) >= float(config["superShare"])
            and (rift is None or time - float(rift["time"]) >= float(config["endingQuietMyr"]))
        )
    if stage == "breakup":
        formed = last_of(EventKind["SUPERCONTINENT"])
        if formed is None:
            return False
        start = float(formed["time"]) - float(config["cycleHoldMyr"])
        first = next(
            (event for event in events
             if event["kind"] == EventKind["RIFT"]
             and start - 1e-9 <= float(event["time"]) <= time + 1e-9),
            None,
        )
        if first is None:
            return False
        age = time - float(first["time"])
        low, high = map(float, config["endingBreakupMyr"])
        return age >= low - 1e-9 and age <= high + 1e-9
    if stage == "dispersed":
        return (
            cycle["state"] == "dispersed"
            and float(cycle["share"]) <= float(config["endingDispersedShare"])
        )
    raise ValueError(f"unknown ending: {stage}")


def history_ending(min_myr: float, stage: str | None, config: dict) -> dict:
    """Return the maximum run horizon and optional stop predicate for an ending."""
    if not stage or stage == "none":
        return {"untilMyr": min_myr, "stopAt": None}
    return {
        "untilMyr": min_myr + float(config["endingExtraMyr"]),
        "stopAt": lambda world: (
            float(world.time) >= min_myr - 1e-9
            and in_stage(world, stage, config)
        ),
    }


def frames_ending(frames, min_myr: float, stage: str | None, config: dict, complete=False):
    """Trim frames to the first eligible ending frame; return None while more history is needed."""
    if not stage or stage == "none":
        if not frames:
            return None
        last = next((i for i, frame in enumerate(frames) if float(frame.time) > min_myr + 1e-9), -1)
        if last < 0:
            return frames if float(frames[-1].time) >= min_myr - 1e-9 else None
        return frames[:last]
    index = next(
        (i for i, frame in enumerate(frames)
         if float(frame.time) >= min_myr - 1e-9 and in_stage(frame, stage, config)),
        -1,
    )
    if index >= 0:
        return frames[:index + 1]
    return frames if complete else None
