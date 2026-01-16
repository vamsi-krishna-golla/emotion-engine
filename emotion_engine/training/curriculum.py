"""
Curriculum Learning

Manages progressive training through increasingly complex scenarios.
"""

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import numpy as np

from emotion_engine.environment.base_env import BaseEmotionEnv
from emotion_engine.environment.scenarios.caretaking import CaretakingScenario
from emotion_engine.environment.scenarios.crisis import CrisisScenario


@dataclass
class CurriculumStage:
    """Configuration for a curriculum stage."""
    name: str
    env_factory: Callable[[], BaseEmotionEnv]
    duration_steps: int
    success_threshold: float = 0.7
    description: str = ""


class CurriculumScheduler:
    """
    Manages curriculum learning schedule.

    Progresses through stages:
    1. Basic Attachment: Learn proximity and basic caregiving
    2. Caretaking: Develop strong attachment and protective behaviors
    3. Crisis: Learn self-sacrifice in dangerous situations
    """

    def __init__(
        self,
        stages: Optional[List[CurriculumStage]] = None,
    ):
        """
        Initialize curriculum scheduler.

        Args:
            stages: List of curriculum stages (if None, uses default)
        """
        if stages is None:
            stages = self._create_default_curriculum()

        self.stages = stages
        self.current_stage_idx = 0
        self.current_stage_steps = 0
        self.stage_performance_history = []

    def _create_default_curriculum(self) -> List[CurriculumStage]:
        """Create default 5-stage curriculum."""
        return [
            CurriculumStage(
                name="Stage 1: Basic Attachment",
                env_factory=lambda: CaretakingScenario(
                    num_children=1,
                    max_steps=500,
                    grid_size=10,
                    child_vulnerability=0.5,
                ),
                duration_steps=1_000_000,
                success_threshold=0.6,
                description="Learn proximity and basic interaction with child agent",
            ),
            CurriculumStage(
                name="Stage 2: Intensive Caretaking",
                env_factory=lambda: CaretakingScenario(
                    num_children=1,
                    max_steps=750,
                    grid_size=10,
                    child_vulnerability=0.7,
                ),
                duration_steps=2_000_000,
                success_threshold=0.7,
                description="Develop strong attachment through caregiving",
            ),
            CurriculumStage(
                name="Stage 3: Multiple Children",
                env_factory=lambda: CaretakingScenario(
                    num_children=2,
                    max_steps=750,
                    grid_size=12,
                    child_vulnerability=0.6,
                ),
                duration_steps=1_500_000,
                success_threshold=0.6,
                description="Manage care for multiple vulnerable agents",
            ),
            CurriculumStage(
                name="Stage 4: Crisis Introduction",
                env_factory=lambda: CrisisScenario(
                    num_agents=2,
                    max_steps=800,
                    grid_size=10,
                    threat_frequency=150,
                    threat_damage=0.2,
                ),
                duration_steps=2_000_000,
                success_threshold=0.5,
                description="Face occasional threats requiring protection",
            ),
            CurriculumStage(
                name="Stage 5: Self-Sacrifice",
                env_factory=lambda: CrisisScenario(
                    num_agents=2,
                    max_steps=1000,
                    grid_size=10,
                    threat_frequency=100,
                    threat_damage=0.3,
                ),
                duration_steps=3_000_000,
                success_threshold=0.4,
                description="Frequent crises testing self-sacrifice behaviors",
            ),
        ]

    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        return self.stages[self.current_stage_idx]

    def get_current_env(self) -> BaseEmotionEnv:
        """Get environment for current stage."""
        return self.get_current_stage().env_factory()

    def step(self, performance_metrics: Dict[str, float]) -> bool:
        """
        Progress curriculum based on performance.

        Args:
            performance_metrics: Metrics from recent training

        Returns:
            True if stage changed, False otherwise
        """
        self.current_stage_steps += 1
        self.stage_performance_history.append(performance_metrics)

        current_stage = self.get_current_stage()

        # Check if we should advance
        if self.current_stage_steps >= current_stage.duration_steps:
            # Check performance threshold
            if len(self.stage_performance_history) >= 10:
                recent_performance = np.mean([
                    m.get('avg_episode_reward', 0)
                    for m in self.stage_performance_history[-10:]
                ])

                if recent_performance >= current_stage.success_threshold:
                    return self._advance_stage()
                else:
                    print(f"  Stage {self.current_stage_idx + 1} performance below threshold "
                          f"({recent_performance:.3f} < {current_stage.success_threshold:.3f})")
                    print(f"  Continuing current stage...")
                    # Reset step count but stay in current stage
                    self.current_stage_steps = 0
                    return False
            else:
                # Not enough data, advance anyway
                return self._advance_stage()

        return False

    def _advance_stage(self) -> bool:
        """Advance to next curriculum stage."""
        if self.current_stage_idx < len(self.stages) - 1:
            old_stage = self.get_current_stage()
            self.current_stage_idx += 1
            new_stage = self.get_current_stage()

            print(f"\n{'='*70}")
            print(f" CURRICULUM ADVANCEMENT")
            print(f"{'='*70}")
            print(f"Completed: {old_stage.name}")
            print(f"Starting:  {new_stage.name}")
            print(f"Description: {new_stage.description}")
            print(f"Target steps: {new_stage.duration_steps:,}")
            print(f"{'='*70}\n")

            self.current_stage_steps = 0
            self.stage_performance_history = []
            return True

        return False

    def get_progress(self) -> Dict[str, any]:
        """Get curriculum progress information."""
        current_stage = self.get_current_stage()

        return {
            'stage_idx': self.current_stage_idx,
            'stage_name': current_stage.name,
            'stage_steps': self.current_stage_steps,
            'stage_duration': current_stage.duration_steps,
            'stage_progress': self.current_stage_steps / current_stage.duration_steps,
            'total_stages': len(self.stages),
            'is_final_stage': self.current_stage_idx == len(self.stages) - 1,
        }

    def is_complete(self) -> bool:
        """Check if curriculum is complete."""
        return (
            self.current_stage_idx == len(self.stages) - 1 and
            self.current_stage_steps >= self.get_current_stage().duration_steps
        )


def create_simple_curriculum() -> CurriculumScheduler:
    """Create a simplified 2-stage curriculum for testing."""
    stages = [
        CurriculumStage(
            name="Stage 1: Basic Caretaking",
            env_factory=lambda: CaretakingScenario(
                num_children=1,
                max_steps=500,
                child_vulnerability=0.6,
            ),
            duration_steps=100_000,
            success_threshold=0.5,
            description="Learn basic parent-child attachment",
        ),
        CurriculumStage(
            name="Stage 2: Crisis Response",
            env_factory=lambda: CrisisScenario(
                num_agents=2,
                max_steps=600,
                threat_frequency=120,
                threat_damage=0.25,
            ),
            duration_steps=100_000,
            success_threshold=0.3,
            description="Learn protective self-sacrifice",
        ),
    ]

    return CurriculumScheduler(stages=stages)
