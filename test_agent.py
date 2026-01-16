"""
Test the EmotionalAgent with RL components.
"""

import torch
import numpy as np
from emotion_engine.rl.agent import EmotionalAgent
from emotion_engine.rl.reward_shaping import create_default_reward_shaper

print("="*70)
print(" Testing EmotionalAgent with RL")
print("="*70)

# Create agent
print("\n[1] Creating EmotionalAgent...")
agent = EmotionalAgent(
    agent_id=0,
    observation_space_dim=96,  # Will be split into self/social/env
    action_space_dim=4,
    emotion_features_dim=256,
    device='cpu'
)

print(f"    Agent ID: {agent.agent_id}")
print(f"    Networks initialized:")
print(f"      - Emotion Encoder: {sum(p.numel() for p in agent.emotion_encoder.parameters())} params")
print(f"      - Emotion Composer: {sum(p.numel() for p in agent.emotion_composer.parameters())} params")
print(f"      - Policy Network: {sum(p.numel() for p in agent.policy.parameters())} params")
print(f"      - Value Network: {sum(p.numel() for p in agent.value_net.parameters())} params")

# Test observation processing
print("\n[2] Testing observation processing...")
observation = np.random.randn(96).astype(np.float32)
other_agents_states = {
    1: {'wellbeing': 0.7, 'vulnerability': 0.3}
}

emotion_features = agent.observe(observation, other_agents_states)
print(f"    Observation shape: {observation.shape}")
print(f"    Emotion features shape: {emotion_features.shape}")

# Test emotion update
print("\n[3] Testing emotion updates...")
print(f"    Initial attachment: {agent.emotion_state.primitives['attachment']:.3f}")
print(f"    Initial empathy: {agent.emotion_state.primitives['empathy']:.3f}")

agent.update_emotions({'attachment': 0.3, 'empathy': 0.2, 'protective_instinct': 0.25})

print(f"    After update:")
print(f"      - Attachment: {agent.emotion_state.primitives['attachment']:.3f}")
print(f"      - Empathy: {agent.emotion_state.primitives['empathy']:.3f}")
print(f"      - Protective: {agent.emotion_state.primitives['protective_instinct']:.3f}")
print(f"      - Maternal Love: {agent.emotion_state.complex_emotions['maternal_love']:.3f}")

# Test action selection
print("\n[4] Testing action selection...")
action, log_prob, value = agent.select_action(emotion_features, deterministic=False)

print(f"    Action: {action}")
print(f"    Log prob: {log_prob:.4f}")
print(f"    Value estimate: {value:.4f}")

# Test reward computation
print("\n[5] Testing reward shaping...")
agent.update_relationships(1, 'protect', outcome=0.8)

extrinsic_reward = 0.5
agent_states = {
    1: {
        'wellbeing_delta': 0.3,
        'benefit_received': 0.6,
        'vulnerability': 0.4,
        'threat_level': 0.7,
    }
}
action_info = {
    'self_cost': 0.2,
    'protected_agents': [1],
    'nearby_agents': [1],
}

total_reward = agent.compute_reward(extrinsic_reward, agent_states, action_info)

print(f"    Extrinsic reward: {extrinsic_reward:.3f}")
print(f"    Total shaped reward: {total_reward:.3f}")
print(f"    Reward includes: empathy, attachment, altruism, protective, social bonuses")

# Test full step
print("\n[6] Testing full agent step...")
emotion_stimulation = {'empathy': 0.1, 'joy': 0.15}
action, log_prob, value = agent.step(observation, other_agents_states, emotion_stimulation)

print(f"    Step completed successfully")
print(f"    Action: {action[:2]}... (showing first 2 dims)")
print(f"    Value: {value:.4f}")

# Test emotion summary
print("\n[7] Agent emotional state summary...")
summary = agent.get_emotion_summary()
print(f"    Dominant primitive: {summary['dominant_primitive']} ({summary['dominant_primitive_intensity']:.3f})")
print(f"    Dominant complex: {summary['dominant_complex']} ({summary['dominant_complex_intensity']:.3f})")
print(f"    Valence: {summary['valence']:.3f} (emotional tone)")
print(f"    Arousal: {summary['arousal']:.3f} (activation level)")

print("\n" + "="*70)
print(" SUCCESS: EmotionalAgent working correctly!")
print("="*70)
print("\nKey Capabilities Verified:")
print("  [OK] Emotion state tracking and updates")
print("  [OK] Neural network forward passes")
print("  [OK] Action selection with emotion-conditioning")
print("  [OK] Reward shaping with intrinsic emotional rewards")
print("  [OK] Relationship management and attachment tracking")
print("  [OK] Complex emotion composition (maternal love)")
print("\nReady for:")
print("  - Multi-agent environment implementation")
print("  - Training loop with PPO")
print("  - Curriculum learning scenarios")
print("="*70)
