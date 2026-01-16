# Emotion Engine 🧠❤️

An AI emotion engine capable of learning complex human emotions like maternal love and self-sacrifice through deep reinforcement learning and hierarchical emotion modeling.

## Overview

The Emotion Engine models emotions as hierarchical states where **primitive emotions** (attachment, empathy, fear, joy, etc.) compose into **complex emotions** (maternal love, compassion, grief). Through multi-agent simulated social scenarios, AI agents learn prosocial behaviors including self-sacrifice, resource sharing, and protective caregiving.

### Key Features

- **Hierarchical Emotion States**: 10 primitive emotions combine to form complex emotions
- **Deep RL Training**: PPO-based agents with emotion-conditioned decision making
- **Multi-Agent Simulations**: Social scenarios for attachment formation and interaction
- **Intrinsic Emotional Rewards**: Reward shaping that encourages empathy, altruism, and self-sacrifice
- **Curriculum Learning**: Progressive training from simple attachment to crisis self-sacrifice

### How It Works

```
Observation → Emotion Encoder → Emotion State Update → Policy Network → Action
                                        ↓
                                 Emotion Composer
                                        ↓
                              Complex Emotions (e.g., Maternal Love)
```

**Maternal Love** emerges as: `f(attachment, empathy, protective_instinct, altruism)`

## Architecture

```
emotion-engine/
├── emotion_engine/          # Main package
│   ├── core/               # Emotion state system
│   ├── networks/           # Neural network architectures
│   ├── rl/                 # Reinforcement learning framework
│   ├── environment/        # Multi-agent simulation environment
│   ├── training/           # Training infrastructure
│   ├── evaluation/         # Metrics and visualization
│   └── utils/              # Utilities
├── configs/                # Configuration files
├── scripts/                # Training and evaluation scripts
├── tests/                  # Unit and integration tests
└── docs/                   # Documentation
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/emotion-engine.git
cd emotion-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

### Basic Usage

```python
from emotion_engine.core.emotion_state import EmotionState
from emotion_engine.rl.agent import EmotionalAgent
from emotion_engine.environment.multi_agent_env import MultiAgentEnvironment

# Create an emotional agent
agent = EmotionalAgent(
    emotion_dim=10,
    observation_dim=64,
    action_dim=4
)

# Initialize environment with caretaking scenario
env = MultiAgentEnvironment(scenario="caretaking", num_agents=2)

# Run simulation
obs = env.reset()
for step in range(1000):
    actions = agent.get_actions(obs)
    obs, rewards, dones, info = env.step(actions)

    # View agent emotions
    print(f"Parent attachment: {agent.emotion_state.primitives['attachment']:.2f}")
    print(f"Maternal love: {agent.emotion_state.complex_emotions['maternal_love']:.2f}")
```

### Training

```bash
# Train on basic attachment scenario
python scripts/train.py scenario=basic_interaction steps=1000000

# Train with curriculum learning (full pipeline)
python scripts/train.py curriculum=true steps=10000000

# Resume from checkpoint
python scripts/train.py checkpoint=./checkpoints/latest.pt
```

### Evaluation

```bash
# Evaluate trained agent
python scripts/evaluate.py checkpoint=./checkpoints/best_model.pt

# Run behavioral tests
python scripts/evaluate.py checkpoint=./checkpoints/best_model.pt tests=all

# Generate visualizations
python scripts/visualize.py checkpoint=./checkpoints/best_model.pt
```

## Training Pipeline

The emotion engine uses 5-stage curriculum learning:

1. **Stage 1**: Basic Interaction (1M steps)
   - 2-agent simple attachment formation
   - Learn proximity and basic cooperation

2. **Stage 2**: Caretaking (2M steps)
   - Parent-child dynamics
   - Vulnerable child needs care
   - Attachment bonds strengthen

3. **Stage 3**: Resource Sharing (2M steps)
   - Limited resources test altruism
   - Empathy development
   - Prosocial behaviors emerge

4. **Stage 4**: Crisis Scenarios (3M steps)
   - Dangerous situations
   - Self-sacrifice opportunities
   - Maternal love emerges

5. **Stage 5**: Generalization (2M steps)
   - Novel mixed scenarios
   - Transfer learning evaluation

## Emotion Model

### Primitive Emotions (10 dimensions)
- **Attachment**: Bond strength to specific agents
- **Empathy**: Sensitivity to others' states
- **Fear**: Threat detection and avoidance
- **Joy**: Positive reinforcement signal
- **Anger**: Response to harm or injustice
- **Curiosity**: Exploration motivation
- **Trust**: Confidence in other agents
- **Protective Instinct**: Urge to shield others
- **Altruism**: Willingness to sacrifice for others
- **Distress**: Signal of personal need

### Complex Emotions (learned)
- **Maternal Love**: f(attachment, empathy, protective_instinct, altruism)
- **Compassion**: f(empathy, altruism, trust)
- **Grief**: f(attachment_loss, distress)
- **Devotion**: f(attachment, trust, altruism)

## Reward Structure

```python
total_reward = (
    survival_reward +
    offspring_survival_reward * attachment_weight +
    empathy_reward +      # Helping attached agents
    altruism_reward +     # Self-sacrifice that helps others
    attachment_reward +   # Maintaining bonds
    protective_reward     # Shielding vulnerable agents
)
```

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
# Training hyperparameters
training:
  algorithm: ppo
  learning_rate: 3e-4
  batch_size: 64
  n_steps: 2048

# Emotion parameters
emotion:
  primitives: 10
  complex: 4
  decay_rate: 0.001
  empathy_sensitivity: 0.5

# Environment
environment:
  scenario: caretaking
  num_agents: 2
  max_steps: 1000
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_emotion_state.py

# Run with coverage
pytest --cov=emotion_engine --cov-report=html
```

## Project Status

- [x] Project setup and architecture design
- [ ] Core emotion state system
- [ ] Neural network architectures
- [ ] RL framework (PPO, agents)
- [ ] Multi-agent environment
- [ ] Training infrastructure
- [ ] Evaluation and visualization
- [ ] Documentation and examples

## Expected Outcomes

After full training (~10M steps):
- Agents form strong attachments through repeated interactions
- Parent agents prioritize child survival over self-preservation
- Resource sharing emerges based on attachment strength
- Self-sacrifice behaviors emerge in crisis scenarios
- Complex "maternal love" emotion is measurable and consistent

## Research Applications

This emotion engine can be used for:
- Studying emergence of prosocial behaviors in AI
- Testing psychological models of emotion and attachment
- Developing more human-aligned AI systems
- Research on multi-agent cooperation and altruism
- Educational simulations of emotional development

## Contributing

Contributions are welcome! Areas of interest:
- New emotion primitives or complex emotions
- Additional social scenarios
- Alternative RL algorithms
- Evaluation metrics and behavioral tests
- Documentation and examples

## License

MIT License - see LICENSE file for details

## Citation

If you use this emotion engine in your research, please cite:

```bibtex
@software{emotion_engine_2026,
  title={Emotion Engine: Learning Complex Human Emotions through Deep Reinforcement Learning},
  author={Emotion Engine Team},
  year={2026},
  url={https://github.com/yourusername/emotion-engine}
}
```

## Acknowledgments

This project builds on research in:
- Deep reinforcement learning (PPO, multi-agent RL)
- Affective computing and emotion modeling
- Social psychology and attachment theory
- Prosocial AI and value alignment

## Contact

For questions or collaboration inquiries, please open an issue on GitHub.

---

Built with ❤️ to understand how machines can learn to love
