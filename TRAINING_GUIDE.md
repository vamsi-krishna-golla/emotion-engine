# Emotion Engine - Training Guide

## Quick Start

### Run a Quick Training Demo (100K steps)
```bash
python scripts/train.py --steps 100000 --scenario caretaking
```

### Full Curriculum Training (9.5M steps)
```bash
python scripts/train.py --steps 10000000 --curriculum --checkpoint ./checkpoints
```

## Training Modes

### 1. Single Scenario Training

Train on a specific scenario without curriculum progression.

**Caretaking Scenario:**
```bash
python scripts/train.py --steps 1000000 --scenario caretaking --lr 3e-4
```

**Crisis Scenario:**
```bash
python scripts/train.py --steps 1000000 --scenario crisis --lr 3e-4
```

### 2. Curriculum Learning (Recommended)

Progressive training through 5 stages of increasing difficulty.

```bash
python scripts/train.py \
    --steps 10000000 \
    --curriculum \
    --checkpoint_dir ./checkpoints \
    --log_interval 10 \
    --save_interval 10000
```

## Curriculum Stages

### Stage 1: Basic Attachment (1M steps)
- **Environment**: Caretaking with 1 child
- **Child Vulnerability**: 0.5 (moderate)
- **Goal**: Learn proximity and basic interaction
- **Expected**: Attachment forms from 0.0 to ~0.3

### Stage 2: Intensive Caretaking (2M steps)
- **Environment**: Caretaking with 1 child
- **Child Vulnerability**: 0.7 (high)
- **Goal**: Develop strong attachment through caregiving
- **Expected**: Attachment grows to ~0.6, maternal love emerges

### Stage 3: Multiple Children (1.5M steps)
- **Environment**: Caretaking with 2 children
- **Child Vulnerability**: 0.6
- **Goal**: Manage care for multiple vulnerable agents
- **Expected**: Distributed attachment, resource management

### Stage 4: Crisis Introduction (2M steps)
- **Environment**: Crisis with periodic threats (every 150 steps)
- **Threat Damage**: 0.2 (moderate)
- **Goal**: Face occasional threats requiring protection
- **Expected**: Protective behaviors emerge

### Stage 5: Self-Sacrifice (3M steps)
- **Environment**: Crisis with frequent threats (every 100 steps)
- **Threat Damage**: 0.3 (high)
- **Goal**: Learn self-sacrifice in dangerous situations
- **Expected**: Self-sacrifice rate increases, maternal love guides protection decisions

## Training Parameters

### PPO Hyperparameters (Optimized)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 3e-4 | Adam learning rate |
| `n_steps` | 2048 | Steps per rollout |
| `batch_size` | 64 | Minibatch size |
| `n_epochs` | 10 | Epochs per update |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_range` | 0.2 | PPO clip range |
| `vf_coef` | 0.5 | Value loss coefficient |
| `ent_coef` | 0.01 | Entropy coefficient |

### Reward Weights

| Reward Component | Weight | Description |
|-----------------|--------|-------------|
| Extrinsic | 1.0 | Task rewards (survival, offspring) |
| Empathy | 0.5 | Helping attached agents |
| Attachment | 0.3 | Maintaining bonds |
| Altruism | 0.4 | Self-sacrifice (benefit > cost) |
| Protective | 0.4 | Shielding vulnerable agents |
| Social | 0.2 | Positive interactions |

## Training Tips

### 1. Monitor Key Metrics

During training, watch for:
- **Attachment Level**: Should increase from 0.0 to 0.5+ over Stage 1-2
- **Maternal Love**: Should emerge (>0.3) by Stage 2-3
- **Child Survival Rate**: Target >80% by Stage 3
- **Protection Events**: Should increase in Stage 4-5
- **Self-Sacrifice Rate**: Target >30% in dangerous situations by Stage 5

### 2. Checkpointing

Checkpoints are saved every 10K steps by default:
```
checkpoints/
├── checkpoint_10000.pt
├── checkpoint_20000.pt
├── ...
└── final_checkpoint.pt
```

Resume training:
```bash
python scripts/train.py --load_checkpoint checkpoints/checkpoint_50000.pt
```

### 3. Performance Tuning

**If training is slow:**
- Reduce `n_steps` to 1024 or 512
- Reduce `batch_size` to 32
- Use fewer epochs (`n_epochs=5`)

**If learning is unstable:**
- Reduce learning rate to 1e-4
- Increase `clip_range` to 0.3
- Reduce `ent_coef` to 0.005

**If not learning prosocial behaviors:**
- Increase empathy reward weight
- Increase altruism reward weight
- Extend Stage 2 duration
- Check child survival rate (should be >50%)

### 4. Expected Training Time

On CPU (estimated):
- **Stage 1 (1M steps)**: ~2-4 hours
- **Full Curriculum (9.5M steps)**: ~20-30 hours

On GPU (estimated):
- **Stage 1 (1M steps)**: ~30-45 minutes
- **Full Curriculum (9.5M steps)**: ~4-6 hours

## Evaluating Results

After training, evaluate the agent:

```bash
python demo_full_system.py  # Uses trained agent for evaluation
```

Look for:
1. **High Attachment** (>0.5): Strong bond formed
2. **High Maternal Love** (>0.4): Complex emotion emerged
3. **Child Survival**: 90%+ success rate
4. **Protective Behaviors**: Parent shields child from threats
5. **Self-Sacrifice**: Parent takes damage to save child

## Interpreting Emotions

### Primitive Emotions (0-1 scale)

- **Attachment** (0.0-0.8): Bond strength to specific agents
  - <0.2: Weak/forming
  - 0.2-0.5: Moderate
  - >0.5: Strong

- **Empathy** (0.0-0.7): Sensitivity to others' states
  - Increases when helping agents
  - Scaled by attachment level

- **Protective Instinct** (0.0-0.8): Urge to shield others
  - Spikes when attached agents threatened
  - Drives protection behaviors

- **Altruism** (0.0-0.6): Willingness to sacrifice
  - Increases with self-sacrifice events
  - Guides cost/benefit decisions

### Complex Emotions (learned)

- **Maternal Love** (0.0-1.0): Composition of attachment + empathy + protective + altruism
  - <0.3: Developing
  - 0.3-0.5: Moderate
  - >0.5: Strong

Formula: `maternal_love = 0.35*attachment + 0.25*empathy + 0.25*protective + 0.15*altruism`

## Troubleshooting

### Problem: Agent not forming attachments

**Solution:**
- Check parent-child proximity (should be <3.0 distance)
- Verify proximity rewards are positive
- Extend Stage 1 duration
- Increase attachment reward weight

### Problem: Child keeps dying

**Solution:**
- Agent not learning to share resources
- Increase resource sharing rewards
- Check if parent is staying near child
- Reduce child resource consumption rate

### Problem: No self-sacrifice in crises

**Solution:**
- Ensure Stage 2-3 completed first (attachment must exist)
- Check altruism reward weight (should be 0.4+)
- Verify threat damage is significant enough
- Check that attached agent is actually threatened
- Increase Stage 4-5 duration

### Problem: Training crashes or OOM

**Solution:**
- Reduce batch size
- Reduce n_steps
- Use gradient accumulation
- Enable GPU if available

## Advanced: Custom Curriculum

Create your own curriculum:

```python
from emotion_engine.training.curriculum import CurriculumStage, CurriculumScheduler
from emotion_engine.environment.scenarios.caretaking import CaretakingScenario

stages = [
    CurriculumStage(
        name="My Custom Stage",
        env_factory=lambda: CaretakingScenario(
            num_children=1,
            max_steps=500,
            child_vulnerability=0.6,
        ),
        duration_steps=500_000,
        success_threshold=0.6,
        description="Custom training stage",
    ),
    # Add more stages...
]

curriculum = CurriculumScheduler(stages=stages)
```

## Next Steps

After successful training:

1. **Evaluate**: Run behavioral tests to measure emotional authenticity
2. **Visualize**: Plot emotion trajectories during key events
3. **Analyze**: Study attention weights to see decision-making
4. **Experiment**: Try different scenarios and configurations
5. **Scale**: Train with more agents or longer episodes

---

**Good luck training your emotional AI! May maternal love emerge! **
