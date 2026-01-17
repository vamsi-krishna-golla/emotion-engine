# Deployment Guide - Interactive Web Demo

This guide shows you how to share the Emotion Engine with the public through an interactive web interface.

## 🚀 Quick Start (Local Demo)

### 1. Install Gradio

```bash
pip install gradio pillow
```

### 2. Launch the Demo

```bash
python web_demo.py
```

This will:
- Start a local web server on port 7860
- Create a **public shareable link** (e.g., `https://xxxxx.gradio.live`)
- Open the interface in your browser automatically

### 3. Share the Link

The Gradio link (shown in console) can be shared with anyone! It's valid for 72 hours.

```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live  ← Share this!
```

## 🌐 Deployment Options

### Option 1: Hugging Face Spaces (Recommended - Free & Easy)

**Best for:** Public demos, no maintenance, free hosting

1. **Create Account**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Sign up for free account

2. **Create New Space**
   - Click "Create new Space"
   - Name: `emotion-engine-demo`
   - SDK: `Gradio`
   - Hardware: `CPU basic` (free tier)

3. **Upload Files**

   Create these files in your Space:

   **`app.py`** (rename web_demo.py):
   ```bash
   # Just copy web_demo.py content
   ```

   **`requirements.txt`**:
   ```
   torch>=2.1.0
   gymnasium>=0.29.0
   numpy>=1.24.0
   gradio>=4.0.0
   pillow>=10.0.0
   matplotlib>=3.8.0
   ```

   **Upload entire `emotion_engine/` directory**

4. **Launch**
   - Space will auto-build and deploy
   - Get permanent URL: `https://huggingface.co/spaces/your-username/emotion-engine-demo`
   - Free, always-on, unlimited visitors!

### Option 2: Gradio Share Link (Easiest - Temporary)

**Best for:** Quick sharing, testing, temporary demos

```bash
python web_demo.py
```

**Pros:**
- Instant deployment (no setup)
- Free shareable link
- Works from your local machine

**Cons:**
- Link expires after 72 hours
- Requires your computer to stay running
- Limited to Gradio's bandwidth

### Option 3: Cloud Platform (DigitalOcean, AWS, etc.)

**Best for:** Permanent hosting, custom domain, full control

#### DigitalOcean Example:

1. **Create Droplet**
   - Ubuntu 22.04
   - Basic plan ($6/month)
   - Add SSH key

2. **Setup Server**
   ```bash
   ssh root@your-server-ip

   # Install Python
   apt update
   apt install python3-pip python3-venv git -y

   # Clone repo
   git clone https://github.com/yourusername/emotion-engine.git
   cd emotion-engine

   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Run with Systemd**

   Create `/etc/systemd/system/emotion-engine.service`:
   ```ini
   [Unit]
   Description=Emotion Engine Web Demo
   After=network.target

   [Service]
   User=root
   WorkingDirectory=/root/emotion-engine
   Environment="PATH=/root/emotion-engine/venv/bin"
   ExecStart=/root/emotion-engine/venv/bin/python web_demo.py

   [Install]
   WantedBy=multi-user.target
   ```

   Enable and start:
   ```bash
   systemctl enable emotion-engine
   systemctl start emotion-engine
   systemctl status emotion-engine
   ```

4. **Setup Nginx Reverse Proxy**
   ```bash
   apt install nginx -y
   ```

   Create `/etc/nginx/sites-available/emotion-engine`:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://127.0.0.1:7860;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
       }
   }
   ```

   Enable:
   ```bash
   ln -s /etc/nginx/sites-available/emotion-engine /etc/nginx/sites-enabled/
   nginx -t
   systemctl restart nginx
   ```

5. **Add SSL (Optional)**
   ```bash
   apt install certbot python3-certbot-nginx -y
   certbot --nginx -d your-domain.com
   ```

### Option 4: Railway / Render (Modern Platforms)

**Railway:**
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Deploy
railway init
railway up
```

**Render:**
1. Connect GitHub repo
2. Create new Web Service
3. Build command: `pip install -r requirements.txt && pip install -e .`
4. Start command: `python web_demo.py`
5. Deploy!

## 📦 Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .
RUN pip install -e .

# Expose port
EXPOSE 7860

# Run demo
CMD ["python", "web_demo.py"]
```

Build and run:
```bash
docker build -t emotion-engine .
docker run -p 7860:7860 emotion-engine
```

## 🔧 Configuration Options

### Customize Server Settings

Edit `web_demo.py` at the bottom:

```python
demo.launch(
    server_name="0.0.0.0",    # Allow external access
    server_port=7860,          # Change port if needed
    share=True,                # Create public Gradio link
    show_error=True,           # Show errors in UI
    auth=("username", "pass"), # Add password protection
    favicon_path="logo.png",   # Custom favicon
)
```

### Add Password Protection

```python
demo.launch(
    auth=("admin", "your-secure-password"),
    # Multiple users:
    # auth=[("user1", "pass1"), ("user2", "pass2")]
)
```

### Custom Domain

For Gradio share links, you can't use custom domain. Use HuggingFace Spaces or self-hosting.

## 📊 Performance Considerations

### CPU vs GPU

The demo runs fine on CPU for single-user inference. For multiple concurrent users:

- **CPU**: Good for 1-5 concurrent users
- **GPU**: Better for 10+ concurrent users

To enable GPU:
```python
device='cuda' if torch.cuda.is_available() else 'cpu'
```

### Caching

Demo creates new agents each run. For better performance, cache trained models:

```python
# Load pre-trained checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')
agent.load_state_dict(checkpoint['agents'][0])
```

### Concurrency Limits

Gradio handles multiple users automatically, but you can limit concurrency:

```python
demo.queue(max_size=10)  # Max 10 in queue
demo.launch()
```

## 🎨 Customization Ideas

### Add More Scenarios
```python
scenario = gr.Dropdown(
    choices=[
        "Caretaking (Parent-Child)",
        "Crisis (Self-Sacrifice)",
        "Resource Sharing",  # New!
        "Family Dynamics",   # New!
    ],
    ...
)
```

### Add Pre-trained Models

Let users compare untrained vs trained agents:
```python
model_choice = gr.Radio(
    choices=["Untrained", "100K steps", "1M steps", "Full curriculum"],
    label="Model Version"
)
```

### Real-time Updates

Show live updates during simulation:
```python
def run_episode_live():
    for step in range(num_steps):
        # ... step logic ...
        yield emotion_plot, summary  # Update UI each step
```

## 🌟 Best Practices

### 1. Add Examples

Help users get started:
```python
gr.Examples(
    examples=[
        ["Caretaking (Parent-Child)", 100],
        ["Crisis (Self-Sacrifice)", 200],
    ],
    inputs=[scenario, num_steps]
)
```

### 2. Add Analytics

Track usage:
```python
import gradio as gr
gr.Analytics(enabled=True)  # Built-in Gradio analytics
```

Or use custom:
```python
from datetime import datetime

def log_usage(scenario, steps):
    with open('usage.log', 'a') as f:
        f.write(f"{datetime.now()},{scenario},{steps}\n")
```

### 3. Error Handling

```python
try:
    result = demo_engine.run_episode(scenario, num_steps)
    return result
except Exception as e:
    return None, None, f"Error: {str(e)}"
```

### 4. Loading States

```python
with gr.Column():
    status = gr.Textbox(label="Status", value="Ready")

run_btn.click(
    fn=lambda: "Running...",
    outputs=status
).then(
    fn=run_episode,
    inputs=[scenario, num_steps],
    outputs=[emotion_plot, health_plot, summary]
).then(
    fn=lambda: "Complete!",
    outputs=status
)
```

## 📱 Mobile Optimization

Gradio is mobile-responsive by default, but you can optimize:

```python
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Use gr.Accordion for collapsible sections
    with gr.Accordion("Advanced Settings", open=False):
        # Advanced options here
        pass
```

## 🚀 Next Steps

After deployment:

1. **Share the link** on social media, Reddit, Twitter
2. **Add to README** so visitors can try it
3. **Collect feedback** through the interface
4. **Monitor usage** and optimize based on patterns
5. **Update models** as you train better versions

## 📚 Resources

- [Gradio Documentation](https://gradio.app/docs/)
- [Hugging Face Spaces Guide](https://huggingface.co/docs/hub/spaces)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

## 💡 Pro Tips

1. **Start with Gradio share link** for testing
2. **Move to Hugging Face Spaces** for permanent free hosting
3. **Use custom server** only if you need advanced features
4. **Add trained models** for better demonstrations
5. **Keep episodes short** (50-100 steps) for faster response

---

**Ready to share?**

```bash
python web_demo.py
```

Then share the public link! 🎉
