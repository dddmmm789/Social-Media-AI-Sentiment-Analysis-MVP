# 🤖 My Sentiment Analysis Journey

🎯 I built this project to understand what it really means to train an AI model with my own hands. Turns out, you can create something useful with just 60 carefully chosen training sentences! However, it is not as sophisticated as RoBERTa, and you can play around with both to see the difference.

⚠️ **Important Note**: While this project runs smoothly on local machines, it currently has compatibility issues with Streamlit Cloud due to Python version conflicts. We're working on a fix!

## 💻 Environment Specifications
This project has been tested and runs successfully on:
- macOS Sonoma 23.6.0
- Python 3.9 (recommended) or 3.10
- M1/M2 Mac architecture
- 16GB RAM (recommended)

## 🚀 Detailed Setup Guide

### Prerequisites
1. Install Python 3.9 or 3.10 (3.12 is not supported due to compatibility issues)
2. Install pip (Python package manager)
3. (Recommended) Install git for version control

### Step-by-Step Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Social-Media-AI-Sentiment-Analysis-MVP.git
   cd Social-Media-AI-Sentiment-Analysis-MVP
   ```

2. Create and activate a virtual environment:
   ```bash
   # On macOS/Linux:
   python3 -m venv venv
   source venv/bin/activate

   # On Windows:
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

### Troubleshooting
- If you encounter PyTorch installation issues, try:
  ```bash
  pip3 install torch torchtext --index-url https://download.pytorch.org/whl/cpu
  ```
- If you see "module not found" errors, ensure you're in the virtual environment
- For M1/M2 Macs, you might need to use Rosetta 2 for some packages

## 📦 What's Inside
- 🧠 A simple sentiment analyzer trained on basic examples (check them out in the app!)
- 🚀 Meta's RoBERTa model for comparison (the big guns!)
- 📊 Side-by-side comparison of both models
- 💻 Interactive web interface using Streamlit

## ✨ Why This is Cool
- 🎓 See how a tiny dataset (60 sentences) can create a working model
- 🌟 Compare it with Meta's advanced model trained on millions of tweets
- ⚖️ Understand the trade-offs between simple and complex models
- ⚡ Built in a few hours using [cursor.at](https://cursor.at)

## 🎯 Potential Uses
- 🛍️ Analyze product reviews
- 🎫 Process support tickets
- 📱 Monitor social media sentiment
- 🔄 ... and more with some tweaks!

## 🚀 Learning Highlights
- 🎓 Training your own model is surprisingly accessible
- 📊 Small, focused datasets can be effective for specific use cases
- 🎨 Streamlit makes building AI demos super easy
- 🔍 The gap between simple and advanced models is fascinating

## 🔮 Next Steps
This is an MVP - feel free to:
- 📈 Add more training data
- 🎯 Adapt it for specific industries
- 🔄 Add features like batch processing
- 🧪 Experiment with different models

## 🐛 Known Issues
- ⚠️ Currently not compatible with Streamlit Cloud (Python 3.12 compatibility issues)
- ⚠️ First run may take longer due to downloading required models and embeddings
- ⚠️ Memory usage might be high when loading both models simultaneously

Built with ❤️ using Python, PyTorch, and Streamlit 