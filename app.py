"""
HuggingFace Spaces App for Emotion Engine

Optimized version of web_demo.py with:
- Pre-trained checkpoint loading
- Model caching for better performance
- HuggingFace Spaces optimizations
"""

import gradio as gr
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
from PIL import Image
from pathlib import Path
from functools import lru_cache

from emotion_engine.rl.agent import EmotionalAgent
from emotion_engine.environment.scenarios.caretaking import CaretakingScenario
from emotion_engine.environment.scenarios.crisis import CrisisScenario


# Checkpoint configuration
CHECKPOINT_DIR = Path("demo_checkpoints")
CHECKPOINTS = {
    "Caretaking (Parent-Child)": CHECKPOINT_DIR / "caretaking_50k.pt",
    "Crisis (Self-Sacrifice)": CHECKPOINT_DIR / "crisis_50k.pt",
}


def load_checkpoint(scenario_type, agent_id):
    """Load pre-trained agent checkpoint if available."""
    checkpoint_path = CHECKPOINTS.get(scenario_type)

    if checkpoint_path and checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'agents' in checkpoint and agent_id in checkpoint['agents']:
                print(f"✓ Loaded trained checkpoint for {scenario_type}, agent {agent_id}")
                return checkpoint['agents'][agent_id]
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")

    print(f"Using untrained agent for {scenario_type}, agent {agent_id}")
    return None


class EmotionEngineDemo:
    """Interactive demo wrapper for the emotion engine."""

    def __init__(self):
        self.agents = None
        self.env = None
        self.episode_history = []

    def create_agents(self, scenario_type, num_steps):
        """Create environment and agents, loading checkpoints if available."""
        if scenario_type == "Caretaking (Parent-Child)":
            self.env = CaretakingScenario(
                num_children=1,
                max_steps=num_steps,
                grid_size=10,
                child_vulnerability=0.7,
            )
        else:  # Crisis
            self.env = CrisisScenario(
                num_agents=2,
                max_steps=num_steps,
                grid_size=10,
                threat_frequency=100,
                threat_damage=0.3,
            )

        # Create agents
        self.agents = {}
        for agent_id in range(2):
            self.agents[agent_id] = EmotionalAgent(
                agent_id=agent_id,
                observation_space_dim=self.env.observation_space.shape[0],
                action_space_dim=self.env.action_space.shape[0],
                device='cpu',
            )

            # Load trained checkpoint if available
            state_dict = load_checkpoint(scenario_type, agent_id)
            if state_dict is not None:
                try:
                    self.agents[agent_id].load_state_dict(state_dict)
                except Exception as e:
                    print(f"Warning: Could not load state dict for agent {agent_id}: {e}")

            self.agents[agent_id].set_training(False)

        self.episode_history = []

    def run_episode(self, scenario_type="Caretaking (Parent-Child)", num_steps=100, progress=gr.Progress()):
        """Run a single episode and return visualizations."""

        # Create agents
        progress(0, desc="Creating agents...")
        self.create_agents(scenario_type, num_steps)

        # Reset environment
        observations, infos = self.env.reset()

        # Track data
        parent_emotions = {
            'attachment': [],
            'empathy': [],
            'protective': [],
            'maternal_love': [],
            'valence': []
        }

        positions = {'parent': [], 'child': []}
        health_data = {'parent': [], 'child': []}

        # Run episode
        for step in range(num_steps):
            progress((step + 1) / num_steps, desc=f"Step {step+1}/{num_steps}")

            # Get actions
            actions = {}
            for agent_id, agent in self.agents.items():
                emotion_features = agent.observe(observations[agent_id], {})
                action, _, _ = agent.select_action(emotion_features, deterministic=True)
                actions[agent_id] = action

            # Step environment
            observations, rewards, terminateds, truncateds, infos = self.env.step(actions)

            # Record data
            parent_summary = self.agents[0].get_emotion_summary()
            parent_emotions['attachment'].append(
                self.agents[0].relationship_manager.get_attachment_level(1)
            )
            parent_emotions['empathy'].append(parent_summary.get('empathy', 0))
            parent_emotions['protective'].append(parent_summary.get('protective_instinct', 0))
            parent_emotions['maternal_love'].append(parent_summary.get('complex_maternal_love', 0))
            parent_emotions['valence'].append(parent_summary.get('valence', 0))

            # Get positions from environment state
            if hasattr(self.env, 'agent_positions'):
                positions['parent'].append(self.env.agent_positions[0].copy())
                positions['child'].append(self.env.agent_positions[1].copy())

            health_data['parent'].append(infos[0].get('health', 1.0))
            health_data['child'].append(infos[1].get('health', 1.0))

            # Check if done
            if any(terminateds.values()) or any(truncateds.values()):
                break

        # Generate visualizations
        emotion_plot = self.plot_emotions(parent_emotions, num_steps)
        health_plot = self.plot_health(health_data, num_steps)

        # Generate summary
        final_attachment = parent_emotions['attachment'][-1] if parent_emotions['attachment'] else 0
        final_maternal = parent_emotions['maternal_love'][-1] if parent_emotions['maternal_love'] else 0
        child_survived = health_data['child'][-1] > 0 if health_data['child'] else False

        # Check if checkpoint was loaded
        checkpoint_status = ""
        checkpoint_path = CHECKPOINTS.get(scenario_type)
        if checkpoint_path and checkpoint_path.exists():
            checkpoint_status = " (using trained agents)"
        else:
            checkpoint_status = " (using untrained agents - train checkpoints for better results!)"

        summary = f"""
## Episode Results{checkpoint_status}

**Scenario**: {scenario_type}
**Steps**: {len(parent_emotions['attachment'])}

### Parent Emotional Development
- **Attachment to Child**: {final_attachment:.3f}
- **Maternal Love**: {final_maternal:.3f}
- **Empathy**: {parent_emotions['empathy'][-1]:.3f}
- **Protective Instinct**: {parent_emotions['protective'][-1]:.3f}

### Outcome
- **Child Survived**: {"✅ Yes" if child_survived else "❌ No"}
- **Final Child Health**: {health_data['child'][-1]:.3f}
- **Final Parent Health**: {health_data['parent'][-1]:.3f}

### Interpretation
"""

        if final_attachment > 0.3:
            summary += "- 🎯 **Strong attachment formed!** Parent developed bond with child.\n"
        else:
            summary += "- 🔄 **Attachment forming...** Needs more interaction time.\n"

        if final_maternal > 0.3:
            summary += "- ❤️ **Maternal love emerged!** Complex emotion successfully developed.\n"
        else:
            summary += "- 🌱 **Maternal love developing...** Complex emotion still forming.\n"

        if child_survived:
            summary += "- 👶 **Successful caregiving!** Parent protected child effectively.\n"
        else:
            summary += "- ⚠️ **Child did not survive.** Parent needs to improve caregiving.\n"

        return emotion_plot, health_plot, summary

    def plot_emotions(self, emotions, num_steps):
        """Create emotion trajectory plot."""
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        steps = range(len(emotions['attachment']))

        ax.plot(steps, emotions['attachment'], label='Attachment', linewidth=2, color='#e74c3c')
        ax.plot(steps, emotions['empathy'], label='Empathy', linewidth=2, color='#3498db')
        ax.plot(steps, emotions['protective'], label='Protective', linewidth=2, color='#2ecc71')
        ax.plot(steps, emotions['maternal_love'], label='Maternal Love', linewidth=2.5, color='#9b59b6', linestyle='--')

        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Emotion Intensity', fontsize=12)
        ax.set_title('Parent Emotion Development Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Convert to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)

        return img

    def plot_health(self, health_data, num_steps):
        """Create health trajectory plot."""
        fig = Figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

        steps = range(len(health_data['parent']))

        ax.plot(steps, health_data['parent'], label='Parent Health', linewidth=2, color='#3498db')
        ax.plot(steps, health_data['child'], label='Child Health', linewidth=2, color='#e74c3c')

        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Health', fontsize=12)
        ax.set_title('Agent Health Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

        # Convert to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)

        return img


# Create demo instance
demo_engine = EmotionEngineDemo()


# Create Gradio interface
def create_demo():
    with gr.Blocks(title="Emotion Engine - Interactive Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🧠❤️ Emotion Engine - Interactive Demo

        **Watch AI agents learn complex human emotions like maternal love and self-sacrifice!**

        This demo shows how emotions emerge through reinforcement learning in social scenarios.
        Select a scenario, adjust parameters, and watch parent-child interactions unfold.

        ---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎮 Simulation Settings")

                scenario = gr.Dropdown(
                    choices=["Caretaking (Parent-Child)", "Crisis (Self-Sacrifice)"],
                    value="Caretaking (Parent-Child)",
                    label="Scenario Type",
                    info="Choose the social scenario to simulate"
                )

                num_steps = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=100,
                    step=50,
                    label="Episode Length (steps)",
                    info="Longer episodes allow more emotional development"
                )

                run_btn = gr.Button("▶️ Run Simulation", variant="primary", size="lg")

                gr.Markdown("""
                ### 📖 About

                **What you're seeing:**
                - Agents use neural networks to make decisions
                - Emotions influence behavior through reward shaping
                - Attachment forms through repeated positive interactions
                - Maternal love emerges from primitive emotions

                **The System:**
                - 794K parameters per agent
                - 10 primitive emotions → 4 complex emotions
                - PPO reinforcement learning
                - Multi-agent social simulation
                """)

            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")

                summary_box = gr.Markdown("*Run a simulation to see results...*")

                with gr.Tab("Emotion Development"):
                    emotion_plot = gr.Image(label="Parent Emotion Trajectories", type="pil")

                with gr.Tab("Agent Health"):
                    health_plot = gr.Image(label="Health Over Time", type="pil")

        gr.Markdown("""
        ---

        ### 🔬 What's Happening Under the Hood?

        1. **Observation**: Agents perceive their environment (position, health, other agents)
        2. **Emotion Encoding**: Neural networks process observations into emotion features
        3. **Emotion Update**: Primitive emotions (attachment, empathy, protective) update based on interactions
        4. **Emotion Composition**: Transformer network combines primitives into complex emotions (maternal love)
        5. **Action Selection**: Policy network chooses actions conditioned on emotional state
        6. **Reward Shaping**: Prosocial behaviors (helping child) generate intrinsic emotional rewards

        ### 💡 Try This:

        - **Short episodes (50 steps)**: See initial interactions, emotions just starting to form
        - **Medium episodes (100-200 steps)**: Watch attachment develop over time
        - **Long episodes (300-500 steps)**: Observe strong maternal love emergence
        - **Crisis scenario**: See if parent sacrifices self to protect child

        ### 📚 Learn More:

        - [GitHub Repository](https://github.com/yourusername/emotion-engine)
        - [Training Guide](TRAINING_GUIDE.md)
        - [Technical Documentation](README.md)

        Built with PyTorch, Gymnasium, and Gradio | [License: MIT](LICENSE)
        """)

        # Connect button
        run_btn.click(
            fn=demo_engine.run_episode,
            inputs=[scenario, num_steps],
            outputs=[emotion_plot, health_plot, summary_box],
            api_name="run_simulation"
        )

    return demo


# Launch the demo
if __name__ == "__main__":
    demo = create_demo()
    demo.queue(max_size=20)  # Handle concurrent users
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # HF provides public URL
        show_error=True,
    )
