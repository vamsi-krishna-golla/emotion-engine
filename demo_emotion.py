"""
Simple demo showing the emotion engine in action.
"""

import torch
from emotion_engine.core.emotion_state import EmotionState, create_baseline_emotion_state, simple_maternal_love_composition
from emotion_engine.core.emotion_dynamics import EmotionDynamics, compute_emotion_change_from_event
from emotion_engine.core.relationship import RelationshipManager

print("="*70)
print(" EMOTION ENGINE DEMO: Learning Maternal Love Through Interactions")
print("="*70)

# Create an emotional agent (parent)
print("\n[1] Creating emotional agent (Parent)...")
emotion_state = create_baseline_emotion_state()
dynamics = EmotionDynamics(decay_rate=0.01)
relationship_mgr = RelationshipManager(agent_id=0)

print(f"    Initial state: Attachment={emotion_state.primitives['attachment']:.2f}, "
      f"Empathy={emotion_state.primitives['empathy']:.2f}")

# Simulate parent-child interactions over time
print("\n[2] Simulating parent-child interactions...")
print()

events = [
    ("proximity_to_attached", 0.6, "Parent stays close to child"),
    ("protected", 0.9, "Parent protects child from danger"),
    ("help_received", 0.7, "Child responds positively to care"),
    ("cooperation_success", 0.8, "Parent and child cooperate"),
    ("self_sacrifice", 1.0, "Parent sacrifices for child's safety"),
]

child_id = 1  # Agent ID for the child

for timestep, (event_type, intensity, description) in enumerate(events, 1):
    print(f"--- Timestep {timestep} ---")
    print(f"Event: {description}")

    # Get current attachment level
    attachment_level = relationship_mgr.get_attachment_level(child_id)

    # Compute emotion changes from event
    emotion_changes = compute_emotion_change_from_event(
        event_type,
        event_intensity=intensity,
        agent_id=child_id,
        attachment_level=attachment_level
    )

    # Update emotion state
    emotion_state = dynamics.step(emotion_state, emotion_changes, delta_time=1.0)

    # Update relationship
    relationship_mgr.update_relationship(
        other_agent_id=child_id,
        interaction_type=event_type.split("_")[0],
        outcome=intensity,
        emotion_state=emotion_state.to_vector(),
        delta_time=1.0
    )

    # Calculate maternal love
    maternal_love = simple_maternal_love_composition(emotion_state)
    attachment = relationship_mgr.get_attachment_level(child_id)

    # Display emotional state
    print(f"  Emotions:")
    print(f"    - Attachment to child: {attachment:.3f}")
    print(f"    - Empathy:             {emotion_state.primitives['empathy']:.3f}")
    print(f"    - Protective Instinct: {emotion_state.primitives['protective_instinct']:.3f}")
    print(f"    - Altruism:            {emotion_state.primitives['altruism']:.3f}")
    print(f"  => MATERNAL LOVE:        {maternal_love:.3f}")
    print()

# Final summary
print("="*70)
print(" RESULTS: Maternal Love Emerged!")
print("="*70)
print()

final_maternal_love = simple_maternal_love_composition(emotion_state)
final_attachment = relationship_mgr.get_attachment_level(child_id)

print(f"Final Emotional State:")
print(f"  - Attachment to child:    {final_attachment:.3f} (started at 0.0)")
print(f"  - Maternal Love:          {final_maternal_love:.3f} (started at ~0.05)")
print(f"  - Valence (mood):         {emotion_state.valence:.3f} (positive)")
print(f"  - Arousal (activation):   {emotion_state.arousal:.3f}")
print()

print("Interpretation:")
if final_maternal_love > 0.5:
    print("  [SUCCESS] Strong maternal love developed through caregiving interactions!")
    print("  The agent now prioritizes the child's wellbeing and shows prosocial behavior.")
elif final_maternal_love > 0.3:
    print("  [GOOD] Moderate maternal love emerging. More interactions would strengthen it.")
else:
    print("  [DEVELOPING] Maternal love is forming. Continued caregiving will develop it.")

print()
print("="*70)
print(" Next Steps:")
print("  - Implement RL framework to train agents through reinforcement learning")
print("  - Create multi-agent environment with crisis scenarios")
print("  - Train neural networks to learn emotion composition")
print("  - Observe self-sacrifice behaviors emerge!")
print("="*70)
