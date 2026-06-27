"""Structured rule keys shared by RuleEngine (producer) and ExplorationPolicy (consumer).

A rule is split into a *condition* (the situation) and an *outcome* (what happened).
The condition is the dict key; outcomes are the stored values. This lets the engine
detect determinism (same condition -> always same outcome?) and lets the policy check
novelty with an exact `condition in hypotheses` lookup instead of fragile string matching.

`movement` and `blocked` are unified into a single ``delta`` condition: a block is just
``delta=(0, 0)``. So "have I tried this action while facing these colors?" is answered by
one condition regardless of whether the result was a move or a block.
"""

from dataclasses import dataclass


def subject(is_actor: bool, obj_profile=None) -> tuple:
    """Identity of the thing a rule is about: the player, or an object shape profile."""
    return ("actor",) if is_actor else ("obj", obj_profile)


@dataclass(frozen=True)
class RuleCondition:
    """Hashable description of a situation (everything before "THEN")."""

    kind: str               # "delta" | "color_change" | "rotation" | "reward"
    action: int
    subject: tuple          # ("actor",) or ("obj", obj_profile)
    cond: tuple = ()        # extra ("name", value) pairs, kind-specific

    def subject_str(self) -> str:
        if self.subject and self.subject[0] == "actor":
            return "actor"
        return f"obj={self.subject[1]}"

    def __str__(self) -> str:
        parts = [f"WHEN action={self.action}", self.subject_str()]
        for name, val in self.cond:
            parts.append(f"{name}={val}")
        return " AND ".join(parts)


def delta_condition(action, is_actor, adj=None, obj_profile=None) -> RuleCondition:
    """Movement/blocked share this. Actor conditions include adjacency; objects don't."""
    cond = (("adj", tuple(sorted(adj))),) if is_actor else ()
    return RuleCondition("delta", action, subject(is_actor, obj_profile), cond)


def color_change_condition(action, is_actor, from_color, obj_profile=None) -> RuleCondition:
    return RuleCondition(
        "color_change", action, subject(is_actor, obj_profile),
        (("from_color", from_color),),
    )


def rotation_condition(action, is_actor, from_rotation, obj_profile=None) -> RuleCondition:
    return RuleCondition(
        "rotation", action, subject(is_actor, obj_profile),
        (("from_rotation", from_rotation),),
    )


def reward_condition(action, is_actor, obj_profile=None) -> RuleCondition:
    return RuleCondition("reward", action, subject(is_actor, obj_profile))


def format_outcome(cond: RuleCondition, outcome) -> str:
    if cond.kind == "delta":
        return f"delta=({outcome[0]},{outcome[1]})"
    if cond.kind == "color_change":
        return f"to_color={outcome}"
    if cond.kind == "rotation":
        return f"to_rotation={outcome}"
    if cond.kind == "reward":
        return f"receive_reward={outcome}"
    return str(outcome)


def format_rule(cond: RuleCondition, outcome) -> str:
    """Readable full sentence for logging/debugging."""
    return f"{cond} THEN {format_outcome(cond, outcome)}"
