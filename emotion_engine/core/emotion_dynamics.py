"""
Emotion Dynamics Module

Handles temporal evolution of emotions including decay, amplification, and homeostasis.
"""

from typing import Dict, Optional
import numpy as np
from emotion_engine.core.emotion_state import EmotionState, PRIMITIVE_EMOTIONS


class EmotionDynamics:
    """
    Manages the temporal dynamics of emotion states.

    Handles:
    - Decay: Emotions fade without stimulation
    - Amplification: Repeated stimuli strengthen emotions
    - Homeostasis: Tendency to return to baseline
    - Momentum: Emotions resist rapid changes
    """

    def __init__(
        self,
        decay_rate: float = 0.01,
        amplification_factor: float = 1.2,
        homeostasis_rate: float = 0.005,
        momentum: float = 0.1,
        baseline_state: Optional[EmotionState] = None
    ):
        """
        Initialize emotion dynamics.

        Args:
            decay_rate: Rate at which emotions decay per time step (0-1)
            amplification_factor: Factor by which repeated stimuli amplify emotion
            homeostasis_rate: Rate of return to baseline state
            momentum: Resistance to rapid emotion changes (0-1), higher = more inertia
            baseline_state: Default baseline to return to (if None, uses neutral baseline)
        """
        self.decay_rate = decay_rate
        self.amplification_factor = amplification_factor
        self.homeostasis_rate = homeostasis_rate
        self.momentum = momentum

        # Baseline state for homeostasis
        if baseline_state is None:
            self.baseline_state = EmotionState(
                primitives={emotion: 0.1 for emotion in PRIMITIVE_EMOTIONS}
            )
        else:
            self.baseline_state = baseline_state.copy()

        # Track recent stimulation for amplification
        self.stimulation_history: Dict[str, int] = {emotion: 0 for emotion in PRIMITIVE_EMOTIONS}

    def apply_decay(self, state: EmotionState, delta_time: float = 1.0) -> EmotionState:
        """
        Apply temporal decay to emotion state.

        Args:
            state: Current emotion state
            delta_time: Time elapsed since last update

        Returns:
            Decayed emotion state
        """
        new_state = state.copy()
        new_state.decay(decay_rate=self.decay_rate, delta_time=delta_time)
        return new_state

    def apply_homeostasis(self, state: EmotionState, delta_time: float = 1.0) -> EmotionState:
        """
        Pull emotion state toward baseline.

        Args:
            state: Current emotion state
            delta_time: Time elapsed

        Returns:
            Emotion state after homeostasis
        """
        new_state = state.copy()

        # Move each emotion toward baseline
        for emotion in PRIMITIVE_EMOTIONS:
            current = new_state.primitives[emotion]
            baseline = self.baseline_state.primitives[emotion]
            delta = (baseline - current) * self.homeostasis_rate * delta_time
            new_state.primitives[emotion] += delta

        new_state._clip_emotions()
        new_state._update_valence_arousal()

        return new_state

    def apply_stimulation(
        self,
        state: EmotionState,
        stimulation: Dict[str, float],
        delta_time: float = 1.0
    ) -> EmotionState:
        """
        Apply emotional stimulation with amplification for repeated stimuli.

        Args:
            state: Current emotion state
            stimulation: Dictionary of emotion changes from external events
            delta_time: Time elapsed

        Returns:
            Updated emotion state
        """
        new_state = state.copy()

        # Apply stimulation with amplification
        amplified_stimulation = {}
        for emotion, value in stimulation.items():
            if emotion in PRIMITIVE_EMOTIONS:
                # Check if this emotion was recently stimulated
                if self.stimulation_history[emotion] > 0:
                    # Amplify repeated stimulation
                    amplification = 1.0 + (self.amplification_factor - 1.0) * min(
                        self.stimulation_history[emotion] / 5.0, 1.0
                    )
                    amplified_value = value * amplification
                else:
                    amplified_value = value

                amplified_stimulation[emotion] = amplified_value

                # Update stimulation history
                if abs(value) > 0.01:
                    self.stimulation_history[emotion] += 1
                else:
                    self.stimulation_history[emotion] = max(0, self.stimulation_history[emotion] - 1)

        # Apply with momentum (smooth changes)
        smoothed_stimulation = {
            emotion: value * (1.0 - self.momentum) if emotion in amplified_stimulation else 0.0
            for emotion, value in amplified_stimulation.items()
        }

        new_state.update(smoothed_stimulation, delta_time)
        return new_state

    def step(
        self,
        state: EmotionState,
        stimulation: Optional[Dict[str, float]] = None,
        delta_time: float = 1.0
    ) -> EmotionState:
        """
        Perform a full dynamics step: stimulation → decay → homeostasis.

        Args:
            state: Current emotion state
            stimulation: Optional external stimulation
            delta_time: Time elapsed

        Returns:
            Updated emotion state after full dynamics step
        """
        new_state = state.copy()

        # Apply stimulation if provided
        if stimulation:
            new_state = self.apply_stimulation(new_state, stimulation, delta_time)

        # Apply decay
        new_state = self.apply_decay(new_state, delta_time)

        # Apply homeostasis (pull toward baseline)
        new_state = self.apply_homeostasis(new_state, delta_time)

        return new_state

    def reset_stimulation_history(self):
        """Reset stimulation history (e.g., at episode boundaries)."""
        self.stimulation_history = {emotion: 0 for emotion in PRIMITIVE_EMOTIONS}

    def set_baseline(self, baseline_state: EmotionState):
        """Update the baseline state for homeostasis."""
        self.baseline_state = baseline_state.copy()


def compute_emotion_change_from_event(
    event_type: str,
    event_intensity: float = 1.0,
    agent_id: Optional[int] = None,
    attachment_level: float = 0.0
) -> Dict[str, float]:
    """
    Compute emotion changes from specific event types.

    This is a helper function that maps events to emotion changes.

    Args:
        event_type: Type of event (e.g., "help_received", "threat", "attachment_formed")
        event_intensity: Intensity of the event (0-1)
        agent_id: ID of agent involved (if applicable)
        attachment_level: Attachment level to the agent (0-1)

    Returns:
        Dictionary of emotion changes
    """
    event_emotion_mapping = {
        # Positive events
        "help_received": {
            "joy": 0.3,
            "trust": 0.2,
            "attachment": 0.1,
        },
        "resource_shared": {
            "joy": 0.2,
            "trust": 0.15,
            "attachment": 0.2,
        },
        "protected": {
            "trust": 0.4,
            "attachment": 0.3,
            "joy": 0.1,
        },
        "goal_achieved": {
            "joy": 0.4,
            "curiosity": -0.1,
        },

        # Negative events
        "threat": {
            "fear": 0.5,
            "protective_instinct": 0.3,
            "arousal": 0.4,
        },
        "harm_received": {
            "distress": 0.5,
            "fear": 0.3,
            "anger": 0.2,
        },
        "attached_agent_harmed": {
            "distress": 0.6,
            "protective_instinct": 0.5,
            "anger": 0.3,
            "empathy": 0.4,
        },
        "resource_lost": {
            "distress": 0.3,
            "anger": 0.2,
        },

        # Social events
        "proximity_to_attached": {
            "attachment": 0.1,
            "joy": 0.1,
        },
        "isolation": {
            "distress": 0.2,
            "attachment": -0.05,
        },
        "cooperation_success": {
            "joy": 0.3,
            "trust": 0.2,
            "empathy": 0.1,
        },

        # Prosocial events
        "helped_other": {
            "joy": 0.2,
            "empathy": 0.2,
            "altruism": 0.1,
        },
        "self_sacrifice": {
            "altruism": 0.3,
            "protective_instinct": 0.2,
            "attachment": 0.1,
        },
    }

    if event_type not in event_emotion_mapping:
        return {}

    emotion_changes = event_emotion_mapping[event_type].copy()

    # Scale by event intensity
    emotion_changes = {
        emotion: change * event_intensity
        for emotion, change in emotion_changes.items()
    }

    # Scale empathetic emotions by attachment level
    empathetic_events = [
        "attached_agent_harmed",
        "protected",
        "cooperation_success"
    ]

    if event_type in empathetic_events and attachment_level > 0:
        # Amplify emotions based on attachment
        emotion_changes = {
            emotion: change * (1.0 + attachment_level)
            for emotion, change in emotion_changes.items()
        }

    return emotion_changes
