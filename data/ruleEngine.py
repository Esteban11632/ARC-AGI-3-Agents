import sys
from collections import defaultdict
from pathlib import Path

_root = Path(__file__).parent.parent
_data = Path(__file__).parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_data) not in sys.path:
    sys.path.insert(0, str(_data))

from utils.environment import get_adjacent_colors, ACTION_TO_DIRECTION


class RuleEngine:
    def __init__(self):
        # Stores history: { rule_key: [list_of_observed_values] }
        self.hypotheses = defaultdict(list)
        # Stores verified rules: { rule_key: deterministic_value }
        self.confirmed_rules = {}
        # Identifies the controllable actor across frames (shape, not color)
        self._actor_profile: tuple | None = None
        self._last_actor_center: tuple[float, float] | None = None
        self._last_actor_track_id: int | None = None
        self.last_actor = None

    def reset_actor(self):
        """Call at the start of a new game — player color/shape may change."""
        self._actor_profile = None
        self._last_actor_center = None
        self._last_actor_track_id = None
        self.last_actor = None

    def _actor_profile_key(self, obj) -> tuple:
        return (obj.shape_type, obj.area, obj.width, obj.height)

    def _center_dist(self, obj) -> float:
        if self._last_actor_center is None:
            return 0.0
        dr = obj.center[0] - self._last_actor_center[0]
        dc = obj.center[1] - self._last_actor_center[1]
        return abs(dr) + abs(dc)

    def _remember_actor(self, obj):
        self._actor_profile = self._actor_profile_key(obj)
        self._last_actor_center = obj.center
        self._last_actor_track_id = obj.matched if obj.matched != -1 else None

    def _find_tracked_actor(self, candidates):
        if self._last_actor_track_id is None:
            return None
        for obj in candidates:
            if obj.matched == self._last_actor_track_id:
                return obj
        return None

    def _leading_edge(self, cells, dr, dc):
        if dc > 0:
            return max(c for _, c in cells)
        if dc < 0:
            return min(c for _, c in cells)
        if dr > 0:
            return max(r for r, _ in cells)
        if dr < 0:
            return min(r for r, _ in cells)
        return 0

    def _blocking_candidates(self, candidates, grid, dr, dc):
        """Objects that did not move but face a non-empty cell in the action direction."""
        blocking = []
        for obj in candidates:
            adj = get_adjacent_colors(obj.cells, grid, dr, dc)
            if not adj or adj == frozenset({0}):
                continue
            blocking.append(obj)
        return blocking

    def _pick_blocked_actor(self, pool, grid, dr, dc):
        """Among non-movers, pick the object first in the path of the action."""
        blocking = self._blocking_candidates(pool, grid, dr, dc)
        if not blocking:
            return None

        reverse = (dr < 0) or (dc < 0)
        edge_key = lambda obj: self._leading_edge(obj.cells, dr, dc)
        return max(blocking, key=edge_key) if reverse else min(blocking, key=edge_key)

    def _get_actor(self, state, next_state, has_direction, dr, dc):
        """Find the object the agent controls — detected by movement, not color."""
        if not has_direction:
            return None

        candidates = [
            obj for obj in next_state.objects
            if not obj.is_large_region and obj.matched != -1
        ]

        if self._last_actor_center is not None:
            profile_matches = [
                obj for obj in candidates
                if self._actor_profile_key(obj) == self._actor_profile
            ]
            if profile_matches:
                tracked = self._find_tracked_actor(profile_matches)
                if tracked is not None:
                    self._remember_actor(tracked)
                    return tracked

                movers = [obj for obj in profile_matches if obj.position_changed]
                if movers:
                    actor = min(movers, key=lambda obj: self._center_dist(obj))
                    self._remember_actor(actor)
                    return actor

                actor = self._pick_blocked_actor(profile_matches, state.grid, dr, dc)
                if actor is not None:
                    self._remember_actor(actor)
                    return actor

                actor = min(profile_matches, key=lambda obj: self._center_dist(obj))
                self._remember_actor(actor)
                return actor

        movers = [obj for obj in candidates if obj.position_changed]
        if movers:
            actor = max(movers, key=lambda obj: obj.area)
            self._remember_actor(actor)
            return actor

        actor = self._pick_blocked_actor(candidates, state.grid, dr, dc)
        if actor is not None:
            self._remember_actor(actor)
            return actor

        if candidates:
            actor = max(candidates, key=lambda obj: obj.area)
            self._remember_actor(actor)
            return actor

        return None

    def analyze_transition(self, transition):
        state = transition.state
        next_state = transition.next_state
        action = transition.action
        reward = transition.reward

        facts = []

        dr, dc = ACTION_TO_DIRECTION.get(action, (0, 0))
        has_direction = (dr, dc) != (0, 0)

        actor = self._get_actor(state, next_state, has_direction, dr, dc)
        self.last_actor = actor

        if actor is not None:
            obj = actor
            adj = get_adjacent_colors(obj.cells, state.grid, dr, dc) if has_direction else frozenset()

            if obj.position_changed:
                facts.append({
                    "type": "movement",
                    "action": action,
                    "shape_type": obj.shape_type,
                    "delta_x": obj.delta_x,
                    "delta_y": obj.delta_y,
                    "adj_colors": adj,
                })

            elif obj.matched != -1 and has_direction:
                facts.append({
                    "type": "blocked",
                    "action": action,
                    "shape_type": obj.shape_type,
                    "adj_colors": adj,
                })

            if obj.color_changed:
                facts.append({
                    "type": "color_change",
                    "action": action,
                    "from_color": obj.previous_color,
                    "to_color": obj.color
                })

            if obj.rotation_changed:
                prev_obj = next(
                    (o for o in state.objects if o.object_id == obj.matched),
                    None,
                )
                facts.append({
                    "type": "rotation",
                    "action": action,
                    "from_rotation": prev_obj.rotation_delta if prev_obj else 0,
                    "to_rotation": obj.rotation_delta,
                    "rotation_delta": obj.rotation_delta
                })

        if reward != 0:
            facts.append({
                "type": "reward",
                "action": action,
                "reward": reward
            })
        
        self._update_hypotheses(facts)
        return facts

    def _update_hypotheses(self, facts):
        """Aggregates facts to separate deterministic rules from random occurrences."""
        for fact in facts:
            fact_type = fact["type"]
            action = fact["action"]

            if fact_type == "movement":
                adj = tuple(sorted(fact["adj_colors"]))
                key = f"WHEN action={action} AND adj={adj} THEN delta=({fact['delta_x']},{fact['delta_y']})"
                value = (fact["delta_x"], fact["delta_y"])
                self.hypotheses[key].append(value)

            elif fact_type == "blocked":
                adj = tuple(sorted(fact["adj_colors"]))
                key = f"WHEN action={action} AND adj={adj} THEN delta=(0,0)"
                value = (0, 0)
                self.hypotheses[key].append(value)

            elif fact_type == "color_change":
                key = f"WHEN action={action} AND initial_color={fact['from_color']} THEN color_shifts_to {fact['to_color']}"
                value = fact["to_color"]
                self.hypotheses[key].append(value)

            elif fact_type == "rotation":
                key = f"WHEN action={action} AND initial_rotation={fact['from_rotation']} THEN rotation_shifts_to {fact['to_rotation']}"
                value = fact["to_rotation"]
                self.hypotheses[key].append(value)

            elif fact_type == "reward":
                key = f"WHEN action={action} THEN receive_reward"
                value = fact["reward"]
                self.hypotheses[key].append(value)

            # Evaluate if this hypothesis has become a stable rule
            self._evaluate_rule_certainty(key)

    def _evaluate_rule_certainty(self, key, threshold=3):
        """Promotes a hypothesis to a confirmed rule if it always yields the same outcome."""
        observations = self.hypotheses[key]
        
        # We need a few samples to rule out pure coincidence
        if len(observations) >= threshold:
            unique_outcomes = set(observations)
            
            # If there is exactly one outcome across all trials, it's a game rule!
            if len(unique_outcomes) == 1:
                self.confirmed_rules[key] = list(unique_outcomes)[0]
            else:
                # If there are multiple outcomes, the rule is conditional (stochastic or environmental)
                self.confirmed_rules.pop(key, None) # Remove if previously added

    def get_rules(self):
        return self.confirmed_rules


if __name__ == "__main__":
    import numpy as np
    from memory import Transition
    from stateBuilder import StateBuilder
    from perceptron import ObjectExtractor

    # Mini grid colors (ls20-like)
    # 0 = background, 2 = player, 8 = floor, 10 = wall

    move_right_steps = [
        (  # (1,1) → (1,2)
            np.array([
                [0, 0, 0, 0, 0, 0],
                [0, 2, 8, 8, 8, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]),
            np.array([
                [0, 0, 0, 0, 0, 0],
                [0, 0, 2, 8, 8, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]),
        ),
        (  # (1,2) → (1,3)
            np.array([
                [0, 0, 0, 0, 0, 0],
                [0, 0, 2, 8, 8, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]),
            np.array([
                [0, 0, 0, 0, 0, 0],
                [0, 0, 8, 2, 8, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]),
        ),
        (  # (1,3) → (1,4)
            np.array([
                [0, 0, 0, 0, 0, 0],
                [0, 0, 8, 2, 8, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]),
            np.array([
                [0, 0, 0, 0, 0, 0],
                [0, 0, 8, 8, 2, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]),
        ),
    ]

    blocked_by_wall = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 2, 10, 8, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ])

    color_change_move = (
        np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 2, 8, 8, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]),
        np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 8, 5, 8, 0],  # player recolored 2 -> 5 while moving right
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]),
    )

    def run_transition(extractor, state_builder, rule_engine, before, after, action):
        extractor.frames.clear()
        state_builder.timestep = 0

        state = state_builder.build_state(before.copy(), None, 0.0)
        next_state = state_builder.build_state(after.copy(), action, 0.0)
        transition = Transition(state, action, 0.0, next_state, False)
        facts = rule_engine.analyze_transition(transition)

        player = rule_engine.last_actor
        print(f"\n--- action={action} ---")
        if player:
            print(
                f"actor color={player.color}, moved={player.position_changed}, "
                f"delta=({player.delta_x},{player.delta_y})"
            )
        for fact in facts:
            if fact["type"] in ("movement", "blocked"):
                print(f"  fact: {fact['type']}, adj={sorted(fact['adj_colors'])}")

        return facts

    extractor = ObjectExtractor(grid_height=6, grid_width=6)
    state_builder = StateBuilder(extractor)
    rule_engine = RuleEngine()
    rule_engine.reset_actor()

    print("=== Test 1: move right on floor (3 steps) ===")
    for before, after in move_right_steps:
        run_transition(extractor, state_builder, rule_engine, before, after, action=4)

    print("\n=== Test 2: blocked by wall (repeat 3x) ===")
    rule_engine.reset_actor()
    for _ in range(3):
        run_transition(extractor, state_builder, rule_engine, blocked_by_wall, blocked_by_wall, action=4)

    print("\n=== Test 3: player recolors while moving ===")
    rule_engine.reset_actor()
    before, after = color_change_move
    facts = run_transition(extractor, state_builder, rule_engine, before, after, action=4)
    player = rule_engine.last_actor
    assert player is not None and player.position_changed, "actor should move despite color change"
    color_facts = [f for f in facts if f["type"] == "color_change"]
    assert color_facts, "should detect in-game color change on tracked actor"
    print(f"  color_change: {color_facts[0]['from_color']} -> {color_facts[0]['to_color']}")
        

    print("\n=== Confirmed rules ===")
    for key, value in rule_engine.get_rules().items():
        print(f"  {key}  ->  {value}")

    print("\n=== All hypotheses (sample) ===")
    for key, values in list(rule_engine.hypotheses.items())[:10]:
        print(f"  {key}: {values}")
