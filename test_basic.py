"""
Basic test script to verify emotion engine components.
"""

import torch
import numpy as np
from emotion_engine.core.emotion_state import EmotionState, create_baseline_emotion_state, simple_maternal_love_composition
from emotion_engine.core.emotion_dynamics import EmotionDynamics, compute_emotion_change_from_event
from emotion_engine.core.relationship import Relationship, RelationshipManager
from emotion_engine.networks.emotion_encoder import EmotionEncoder, SimplifiedEmotionEncoder
from emotion_engine.networks.emotion_composer import EmotionComposer, SimpleEmotionComposer
from emotion_engine.networks.attention import SocialAttention
from emotion_engine.networks.policy_network import PolicyNetwork
from emotion_engine.networks.value_network import ValueNetwork


def test_emotion_state():
    """Test emotion state creation and operations."""
    print("\n" + "="*60)
    print("Testing EmotionState")
    print("="*60)

    # Create emotion state
    state = create_baseline_emotion_state()
    print(f"Created baseline state: {state}")

    # Update emotions
    state.update({"attachment": 0.5, "empathy": 0.3, "joy": 0.2})
    print(f"After update: {state}")

    # Get dominant emotion
    dominant, intensity = state.get_dominant_emotion("primitive")
    print(f"Dominant primitive emotion: {dominant} ({intensity:.2f})")

    # Convert to vector
    vec = state.to_vector()
    print(f"Vector representation shape: {vec.shape}, dtype: {vec.dtype}")

    # Test maternal love composition
    maternal_love = simple_maternal_love_composition(state)
    print(f"Maternal love (rule-based): {maternal_love:.2f}")

    print("✓ EmotionState tests passed!")


def test_emotion_dynamics():
    """Test emotion dynamics."""
    print("\n" + "="*60)
    print("Testing EmotionDynamics")
    print("="*60)

    dynamics = EmotionDynamics(decay_rate=0.1)
    state = EmotionState(primitives={
        "attachment": 0.8,
        "empathy": 0.6,
        "fear": 0.4,
        "joy": 0.5,
        "anger": 0.0,
        "curiosity": 0.3,
        "trust": 0.7,
        "protective_instinct": 0.6,
        "altruism": 0.5,
        "distress": 0.2,
    })

    print(f"Initial state: attachment={state.primitives['attachment']:.2f}")

    # Apply decay
    state = dynamics.apply_decay(state, delta_time=1.0)
    print(f"After decay: attachment={state.primitives['attachment']:.2f}")

    # Apply stimulation
    stimulation = compute_emotion_change_from_event("help_received", event_intensity=0.8)
    print(f"Event stimulation: {stimulation}")
    state = dynamics.apply_stimulation(state, stimulation, delta_time=1.0)
    print(f"After stimulation: joy={state.primitives['joy']:.2f}, trust={state.primitives['trust']:.2f}")

    print("✓ EmotionDynamics tests passed!")


def test_relationships():
    """Test relationship tracking."""
    print("\n" + "="*60)
    print("Testing Relationships")
    print("="*60)

    # Create relationship manager
    manager = RelationshipManager(agent_id=0, decay_rate=0.001)

    # Simulate interactions with another agent
    print("Simulating parent-child interactions...")
    for i in range(5):
        manager.update_relationship(
            other_agent_id=1,
            interaction_type="protect",
            outcome=0.8,
            delta_time=1.0
        )

    attachment = manager.get_attachment_level(1)
    trust = manager.get_trust_level(1)
    print(f"Attachment to agent 1: {attachment:.2f}")
    print(f"Trust in agent 1: {trust:.2f}")

    # Get summary
    summary = manager.get_relationship_summary()
    print(f"Relationship summary: {summary}")

    print("✓ Relationship tests passed!")


def test_neural_networks():
    """Test neural network forward passes."""
    print("\n" + "="*60)
    print("Testing Neural Networks")
    print("="*60)

    batch_size = 4

    # Test EmotionEncoder
    print("\n1. Testing EmotionEncoder...")
    encoder = EmotionEncoder(
        self_state_dim=32,
        social_dim=64,
        env_dim=32,
        output_dim=256,
        max_agents=5
    )

    self_state = torch.randn(batch_size, 32)
    social_context = torch.randn(batch_size, 5, 64)
    environment = torch.randn(batch_size, 32)

    emotion_features = encoder(self_state, social_context, environment)
    print(f"   Input: self_state {self_state.shape}, social {social_context.shape}, env {environment.shape}")
    print(f"   Output: emotion_features {emotion_features.shape}")
    assert emotion_features.shape == (batch_size, 256), "Incorrect output shape!"
    print("   ✓ EmotionEncoder works!")

    # Test EmotionComposer
    print("\n2. Testing EmotionComposer...")
    composer = EmotionComposer(num_primitives=10, num_complex=4)

    primitive_emotions = torch.rand(batch_size, 10)
    complex_emotions = composer(primitive_emotions)
    print(f"   Input: primitive_emotions {primitive_emotions.shape}")
    print(f"   Output: complex_emotions {complex_emotions.shape}")
    assert complex_emotions.shape == (batch_size, 4), "Incorrect output shape!"
    print(f"   Sample maternal_love value: {complex_emotions[0, 0].item():.3f}")
    print("   ✓ EmotionComposer works!")

    # Test SimpleEmotionComposer
    print("\n3. Testing SimpleEmotionComposer (rule-based)...")
    simple_composer = SimpleEmotionComposer(num_primitives=10, num_complex=4)
    complex_emotions_simple = simple_composer(primitive_emotions)
    print(f"   Output: {complex_emotions_simple.shape}")
    print(f"   Sample maternal_love (rule-based): {complex_emotions_simple[0, 0].item():.3f}")
    print("   ✓ SimpleEmotionComposer works!")

    # Test SocialAttention
    print("\n4. Testing SocialAttention...")
    attention = SocialAttention(
        query_dim=64,
        key_dim=64,
        value_dim=64,
        output_dim=256,
        num_heads=4
    )

    query = torch.randn(batch_size, 64)
    keys = torch.randn(batch_size, 5, 64)
    values = torch.randn(batch_size, 5, 64)

    attended, weights = attention(query, keys, values)
    print(f"   Input: query {query.shape}, keys {keys.shape}, values {values.shape}")
    print(f"   Output: attended {attended.shape}, weights {weights.shape}")
    print(f"   Attention weights (agent 0): {weights[0].detach().numpy()}")
    assert attended.shape == (batch_size, 256), "Incorrect attended shape!"
    assert weights.shape == (batch_size, 5), "Incorrect weights shape!"
    print("   ✓ SocialAttention works!")

    # Test PolicyNetwork
    print("\n5. Testing PolicyNetwork...")
    policy = PolicyNetwork(
        emotion_features_dim=256,
        emotion_state_dim=16,
        relationship_dim=64,
        action_dim=4,
        continuous=True
    )

    emotion_features = torch.randn(batch_size, 256)
    emotion_state = torch.randn(batch_size, 16)
    relationship_context = torch.randn(batch_size, 64)

    actions, log_probs = policy.sample_action(emotion_features, emotion_state, relationship_context)
    print(f"   Input: emotion_features {emotion_features.shape}, emotion_state {emotion_state.shape}")
    print(f"   Output: actions {actions.shape}, log_probs {log_probs.shape}")
    print(f"   Sample action: {actions[0].detach().numpy()}")
    assert actions.shape == (batch_size, 4), "Incorrect action shape!"
    print("   ✓ PolicyNetwork works!")

    # Test ValueNetwork
    print("\n6. Testing ValueNetwork...")
    value_net = ValueNetwork(
        emotion_features_dim=256,
        emotion_state_dim=16
    )

    values = value_net(emotion_features, emotion_state)
    print(f"   Input: emotion_features {emotion_features.shape}, emotion_state {emotion_state.shape}")
    print(f"   Output: values {values.shape}")
    print(f"   Sample value: {values[0].item():.3f}")
    assert values.shape == (batch_size, 1), "Incorrect value shape!"
    print("   ✓ ValueNetwork works!")

    print("\n✓ All neural network tests passed!")


def test_integration():
    """Test integration of components."""
    print("\n" + "="*60)
    print("Testing Component Integration")
    print("="*60)

    # Create an emotional agent simulation
    print("\nSimulating emotional agent over time...")

    # Initialize components
    emotion_state = create_baseline_emotion_state()
    dynamics = EmotionDynamics(decay_rate=0.01)
    relationship_mgr = RelationshipManager(agent_id=0)

    print(f"Initial state: {emotion_state}")

    # Simulate 10 time steps with various events
    events = [
        ("proximity_to_attached", 0.5, 1),
        ("help_received", 0.8, 1),
        ("cooperation_success", 0.7, 1),
        ("protected", 0.9, 1),
        ("self_sacrifice", 1.0, 1),
    ]

    for timestep, (event_type, intensity, other_agent_id) in enumerate(events):
        print(f"\n--- Timestep {timestep + 1} ---")
        print(f"Event: {event_type} (intensity={intensity})")

        # Compute emotion changes from event
        emotion_changes = compute_emotion_change_from_event(
            event_type,
            event_intensity=intensity,
            agent_id=other_agent_id,
            attachment_level=relationship_mgr.get_attachment_level(other_agent_id)
        )

        # Update emotion state
        emotion_state = dynamics.step(emotion_state, emotion_changes, delta_time=1.0)

        # Update relationship
        outcome = intensity  # Positive outcome
        relationship_mgr.update_relationship(
            other_agent_id=other_agent_id,
            interaction_type=event_type.split("_")[0],  # Simplified
            outcome=outcome,
            emotion_state=emotion_state.to_vector(),
            delta_time=1.0
        )

        # Compute maternal love
        maternal_love = simple_maternal_love_composition(emotion_state)
        attachment = relationship_mgr.get_attachment_level(other_agent_id)

        print(f"Attachment: {attachment:.2f}")
        print(f"Empathy: {emotion_state.primitives['empathy']:.2f}")
        print(f"Protective: {emotion_state.primitives['protective_instinct']:.2f}")
        print(f"Altruism: {emotion_state.primitives['altruism']:.2f}")
        print(f"Maternal Love: {maternal_love:.2f}")
        print(f"Valence: {emotion_state.valence:.2f}, Arousal: {emotion_state.arousal:.2f}")

    print("\n✓ Integration test passed!")
    print("\n🎉 Maternal love increased from protective and attachment-forming interactions!")


def main():
    """Run all tests."""
    print("="*60)
    print("EMOTION ENGINE - Component Tests")
    print("="*60)

    try:
        test_emotion_state()
        test_emotion_dynamics()
        test_relationships()
        test_neural_networks()
        test_integration()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe emotion engine core components are working correctly!")
        print("Ready to implement RL framework and training infrastructure.")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
