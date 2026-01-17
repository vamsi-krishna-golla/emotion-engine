# 🌐 Interactive Web Demo

## Launch the Demo

```bash
python web_demo.py
```

This launches an interactive web interface where you can:
- ✨ Watch AI agents interact in real-time
- 📊 See emotions develop live (attachment, maternal love, empathy)
- 🎮 Control scenario parameters
- 📈 Visualize emotional trajectories

## What You'll See

The demo runs parent-child scenarios where:
1. **Parent agent** learns to care for vulnerable child
2. **Emotions form** through repeated positive interactions
3. **Maternal love emerges** from primitive emotions
4. **Attachment strengthens** over the episode

## Share with Others

When you run the demo, you get two links:

```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live  ← Share this!
```

The **public URL** can be shared with anyone worldwide!
- Valid for 72 hours
- No setup required
- Works on mobile too

## Features

### Two Scenarios

**Caretaking (Parent-Child)**
- Parent learns to protect vulnerable child
- Emotions: Attachment, empathy, protective instinct
- Watch maternal love emerge!

**Crisis (Self-Sacrifice)**
- Dangerous environment with periodic threats
- Parent must decide: protect self or child?
- See prosocial behaviors in action

### Real-time Visualization

- **Emotion trajectories**: Watch emotions evolve step-by-step
- **Health tracking**: See if parent keeps child alive
- **Summary analysis**: Understand what happened

### Interactive Controls

- Choose scenario type
- Adjust episode length (50-500 steps)
- Run multiple simulations
- Compare results

## Quick Examples

**Short Episode (50 steps)**
- See initial interactions
- Emotions just starting to form
- ~10 seconds to run

**Medium Episode (100-200 steps)**
- Watch attachment develop
- See caregiving behaviors
- ~20-40 seconds to run

**Long Episode (300-500 steps)**
- Strong maternal love emergence
- Complex emotional patterns
- ~1-2 minutes to run

## Technical Details

The demo uses:
- **Gradio** for web interface
- **Matplotlib** for visualizations
- **Untrained agents** (random initialization)
- **CPU inference** (no GPU needed)

For better results, train agents first:
```bash
python scripts/train.py --steps 1000000 --scenario caretaking
```

Then modify `web_demo.py` to load the trained checkpoint.

## Deployment Options

See [DEPLOYMENT.md](DEPLOYMENT.md) for:
- **Hugging Face Spaces** (free permanent hosting)
- **Custom server** (DigitalOcean, AWS, etc.)
- **Docker** deployment
- **Mobile optimization**

## Customization

Edit `web_demo.py` to:
- Add more scenarios
- Change visualization styles
- Add pre-trained models
- Modify UI layout
- Add analytics tracking

## Troubleshooting

**Port already in use?**
```python
demo.launch(server_port=7861)  # Use different port
```

**Want to run in background?**
```bash
nohup python web_demo.py > demo.log 2>&1 &
```

**Need to stop the server?**
- Press `Ctrl+C` in terminal
- Or find process: `pkill -f web_demo.py`

## Next Steps

1. **Launch the demo**: `python web_demo.py`
2. **Share the public link** with friends/colleagues
3. **Try different scenarios** and episode lengths
4. **Deploy permanently** on Hugging Face Spaces (free!)

---

**Ready to share your emotion engine with the world? 🚀**

```bash
python web_demo.py
```
