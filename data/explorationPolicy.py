import random
from utils.environment import get_adjacent_colors, ACTION_TO_DIRECTION
from utils.rule_keys import (
    delta_condition,
    color_change_condition,
    rotation_condition,
    reward_condition,
    format_rule,
)


class ExplorationPolicy:
    def __init__(self, rule_engine, memory, epsilon=0.2, seed=None):
        self.rule_engine = rule_engine
        self.memory = memory
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        # remembers (state_signature, action) we've already tried
        self._tried = set()

    def reset(self):
        """Call on new game / RESET."""
        self._tried.clear()

    # --- helpers ---------------------------------------------------

    def _state_signature(self, state):
        """Cheap, position-aware fingerprint so repeats can be detected."""
        return tuple(
            (o.color, o.center, o.shape_type)
            for o in state.objects if not o.is_large_region
        )

    def _is_known_useless(self, action, adj):
        """Confirmed to do nothing (blocked) => skip."""
        cond = delta_condition(action, is_actor=True, adj=adj)
        return self.rule_engine.confirmed_rules.get(cond) == (0, 0)

    def _is_novel_movement(self, action, adj):
        """This (action, adj) situation has never been observed for the actor."""
        cond = delta_condition(action, is_actor=True, adj=adj)
        return cond not in self.rule_engine.hypotheses

    def _is_novel_color_change(self, action, from_color):
        """This (action, from_color) situation has never been observed for the actor."""
        cond = color_change_condition(action, is_actor=True, from_color=from_color)
        return cond not in self.rule_engine.hypotheses
    
    def _is_novel_rotation(self, action, from_rotation):
        """This (action, from_rotation) situation has never been observed for the actor."""
        cond = rotation_condition(action, is_actor=True, from_rotation=from_rotation)
        return cond not in self.rule_engine.hypotheses

    # --- main entry ------------------------------------------------

    def _is_novel(self, action, adj, actor, tried):
        """True if this action could reveal something unknown about the actor.

        Movement novelty is reliable (a delta fact is recorded every directional
        action). Color/rotation facts are only recorded when a change actually
        happens, so their absence means "never caused that effect", not "untried".
        We therefore only trust color/rotation novelty for situations we have not
        tried yet (otherwise every action would look novel forever).
        """
        if self._is_novel_movement(action, adj):
            return True
        if not tried:
            if self._is_novel_color_change(action, actor.color):
                return True
            if self._is_novel_rotation(action, actor.rotation_delta):
                return True
        return False

    def choose_action(self, state, available_actions):
        actor = self.rule_engine.last_actor
        sig = self._state_signature(state)

        scored = []
        for action in available_actions:
            direction = ACTION_TO_DIRECTION.get(action)

            # non-directional actions: only mild novelty bonus
            if direction is None or actor is None:
                novelty = 0.5 if (sig, action) not in self._tried else 0.0
                scored.append((novelty, action))
                continue

            dr, dc = direction
            adj = get_adjacent_colors(actor.cells, state.grid, dr, dc)
            tried = (sig, action) in self._tried

            if self._is_novel(action, adj, actor, tried):
                score = 1.0                       # explore first (any unknown effect)
            elif self._is_known_useless(action, adj):
                score = -1.0                      # confirmed no-op, avoid
            elif tried:
                score = 0.0                       # already tried here
            else:
                score = 0.3                       # known-ish, still ok
            scored.append((score, action))

        # epsilon-greedy: sometimes pick random legal action
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(available_actions)
        else:
            best = max(s for s, _ in scored)
            action = self.rng.choice([a for s, a in scored if s == best])

        self._tried.add((sig, action))
        return action

    # --- after the env step ----------------------------------------

    def record(self, transition):
        """Store transition and let the rule engine learn from it."""
        self.memory.add(
            transition.state, transition.action,
            transition.reward, transition.next_state, transition.done,
        )
        return self.rule_engine.analyze_transition(transition)