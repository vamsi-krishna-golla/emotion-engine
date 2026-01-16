"""
Emotion State Module

Defines the core emotion state representation with primitive and complex emotions.
This is the foundation of the entire emotion engine.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
import time


# Define primitive emotion names
PRIMITIVE_EMOTIONS = [
    "attachment",
    "empathy",
    "fear",
    "joy",
    "anger",
    "curiosity",
    "trust",
    "protective_instinct",
    "altruism",
    "distress",
]

# Define complex emotion names (learned through composition)
COMPLEX_EMOTIONS = [
    "maternal_love",
    "compassion",
    "grief",
    "devotion",
]


@dataclass
class EmotionState:
    """
    Represents an agent's emotional state at a given moment.

    Emotions are represented as continuous values in [0, 1] where:
    - 0 = emotion not present
    - 1 = maximum intensity

    Attributes:
        primitives: Dictionary of primitive emotion intensities
        complex_emotions: Dictionary of complex emotion intensities (computed/learned)
        valence: Overall positive/negative emotional state (-1 to 1)
        arousal: Overall activation level (0 to 1)
        timestamp: When this emotion state was created/updated
    """

    primitives: Dict[str, float] = field(default_factory=dict)
    complex_emotions: Dict[str, float] = field(default_factory=dict)
    valence: float = 0.0
    arousal: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        """Initialize primitive and complex emotions if not provided."""
        if not self.primitives:
            self.primitives = {emotion: 0.0 for emotion in PRIMITIVE_EMOTIONS}

        if not self.complex_emotions:
            self.complex_emotions = {emotion: 0.0 for emotion in COMPLEX_EMOTIONS}

        # Ensure all values are in valid range
        self._clip_emotions()

        # Calculate valence and arousal if not set
        if self.valence == 0.0 and self.arousal == 0.0:
            self._update_valence_arousal()

    def _clip_emotions(self):
        """Ensure all emotion values are in [0, 1] range."""
        for emotion in self.primitives:
            self.primitives[emotion] = np.clip(self.primitives[emotion], 0.0, 1.0)

        for emotion in self.complex_emotions:
            self.complex_emotions[emotion] = np.clip(self.complex_emotions[emotion], 0.0, 1.0)

        self.valence = np.clip(self.valence, -1.0, 1.0)
        self.arousal = np.clip(self.arousal, 0.0, 1.0)

    def _update_valence_arousal(self):
        """
        Calculate valence (positive/negative) and arousal (activation) from primitive emotions.

        Valence: joy, trust, curiosity contribute positively; fear, anger, distress negatively
        Arousal: fear, anger, joy, protective_instinct increase activation
        """
        # Positive emotions
        positive = (
            self.primitives["joy"] +
            self.primitives["trust"] +
            self.primitives["curiosity"] +
            self.primitives["attachment"] * 0.5
        )

        # Negative emotions
        negative = (
            self.primitives["fear"] +
            self.primitives["anger"] +
            self.primitives["distress"]
        )

        # Valence is the balance between positive and negative
        total_emotion = positive + negative
        if total_emotion > 0:
            self.valence = (positive - negative) / total_emotion
        else:
            self.valence = 0.0

        # Arousal is overall activation level
        self.arousal = np.mean([
            self.primitives["fear"],
            self.primitives["anger"],
            self.primitives["joy"],
            self.primitives["protective_instinct"],
            self.primitives["curiosity"],
        ])

    def update(self, delta_emotions: Dict[str, float], delta_time: float = 1.0):
        """
        Update emotion state with changes.

        Args:
            delta_emotions: Dictionary of emotion changes (can be positive or negative)
            delta_time: Time elapsed since last update (for temporal dynamics)
        """
        # Update primitive emotions
        for emotion, delta in delta_emotions.items():
            if emotion in self.primitives:
                self.primitives[emotion] += delta * delta_time

        # Clip to valid range
        self._clip_emotions()

        # Update valence and arousal
        self._update_valence_arousal()

        # Update timestamp
        self.timestamp = time.time()

    def decay(self, decay_rate: float = 0.01, delta_time: float = 1.0):
        """
        Apply temporal decay to emotions (emotions fade without stimulation).

        Args:
            decay_rate: Rate of decay per time unit
            delta_time: Time elapsed since last decay
        """
        decay_factor = 1.0 - (decay_rate * delta_time)
        decay_factor = max(0.0, decay_factor)

        # Decay primitive emotions toward baseline (0)
        for emotion in self.primitives:
            self.primitives[emotion] *= decay_factor

        # Decay complex emotions
        for emotion in self.complex_emotions:
            self.complex_emotions[emotion] *= decay_factor

        # Update valence and arousal
        self._update_valence_arousal()

    def get_dominant_emotion(self, emotion_type: str = "primitive") -> Tuple[str, float]:
        """
        Get the dominant emotion and its intensity.

        Args:
            emotion_type: "primitive" or "complex"

        Returns:
            Tuple of (emotion_name, intensity)
        """
        if emotion_type == "primitive":
            emotions = self.primitives
        elif emotion_type == "complex":
            emotions = self.complex_emotions
        else:
            raise ValueError(f"Unknown emotion_type: {emotion_type}")

        if not emotions:
            return ("none", 0.0)

        dominant = max(emotions.items(), key=lambda x: x[1])
        return dominant

    def to_vector(self) -> np.ndarray:
        """
        Convert emotion state to a vector representation.

        Returns:
            numpy array of shape (num_primitives + num_complex + 2,)
            [primitive_emotions, complex_emotions, valence, arousal]
        """
        primitive_values = [self.primitives[emotion] for emotion in PRIMITIVE_EMOTIONS]
        complex_values = [self.complex_emotions[emotion] for emotion in COMPLEX_EMOTIONS]

        return np.array(primitive_values + complex_values + [self.valence, self.arousal], dtype=np.float32)

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "EmotionState":
        """
        Create an EmotionState from a vector representation.

        Args:
            vector: numpy array of emotion values

        Returns:
            EmotionState instance
        """
        num_primitives = len(PRIMITIVE_EMOTIONS)
        num_complex = len(COMPLEX_EMOTIONS)

        primitives = {
            PRIMITIVE_EMOTIONS[i]: float(vector[i])
            for i in range(num_primitives)
        }

        complex_emotions = {
            COMPLEX_EMOTIONS[i]: float(vector[num_primitives + i])
            for i in range(num_complex)
        }

        valence = float(vector[num_primitives + num_complex])
        arousal = float(vector[num_primitives + num_complex + 1])

        return cls(
            primitives=primitives,
            complex_emotions=complex_emotions,
            valence=valence,
            arousal=arousal
        )

    def get_dimension(self) -> int:
        """Return the dimensionality of the emotion vector."""
        return len(PRIMITIVE_EMOTIONS) + len(COMPLEX_EMOTIONS) + 2

    def __repr__(self) -> str:
        """String representation of emotion state."""
        dominant_prim, prim_intensity = self.get_dominant_emotion("primitive")
        dominant_comp, comp_intensity = self.get_dominant_emotion("complex")

        return (
            f"EmotionState("
            f"dominant_primitive={dominant_prim}({prim_intensity:.2f}), "
            f"dominant_complex={dominant_comp}({comp_intensity:.2f}), "
            f"valence={self.valence:.2f}, arousal={self.arousal:.2f})"
        )

    def copy(self) -> "EmotionState":
        """Create a deep copy of this emotion state."""
        return EmotionState(
            primitives=self.primitives.copy(),
            complex_emotions=self.complex_emotions.copy(),
            valence=self.valence,
            arousal=self.arousal,
            timestamp=self.timestamp
        )


def create_baseline_emotion_state() -> EmotionState:
    """
    Create a baseline emotion state with neutral values.

    Returns:
        EmotionState with all emotions at low baseline levels
    """
    return EmotionState(
        primitives={emotion: 0.1 for emotion in PRIMITIVE_EMOTIONS},
        complex_emotions={emotion: 0.0 for emotion in COMPLEX_EMOTIONS}
    )


def simple_maternal_love_composition(state: EmotionState) -> float:
    """
    Simple rule-based composition of maternal love from primitive emotions.
    This can serve as a baseline before learning the composition.

    Maternal Love = f(attachment, empathy, protective_instinct, altruism)

    Args:
        state: EmotionState to compute maternal love from

    Returns:
        Maternal love intensity (0-1)
    """
    # Weighted average of relevant primitive emotions
    weights = {
        "attachment": 0.35,
        "empathy": 0.25,
        "protective_instinct": 0.25,
        "altruism": 0.15,
    }

    maternal_love = sum(
        state.primitives[emotion] * weight
        for emotion, weight in weights.items()
    )

    return np.clip(maternal_love, 0.0, 1.0)
