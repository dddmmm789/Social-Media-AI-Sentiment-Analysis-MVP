# Sentiment Analysis Model Comparison: Simple vs Meta's RoBERTa

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sentiment-analysis-comparison.streamlit.app)

This project demonstrates the difference between a simple sentiment analysis model and Meta's state-of-the-art RoBERTa model. It's a perfect example of how deep learning has evolved and the trade-offs between simple and complex models.

## 🌟 Live Demo
Try the live demo at: [sentiment-analysis-comparison.streamlit.app](https://sentiment-analysis-comparison.streamlit.app)

## 🤖 Models Compared

### 1. Simple Model (GloVe + LSTM)
- Custom-trained on 60 examples
- Uses pre-trained GloVe embeddings
- Bidirectional LSTM architecture
- ~100MB model size
- Fast inference

### 2. Meta's RoBERTa
- Pre-trained on 160GB of text
- 355M parameters
- State-of-the-art performance
- Better context understanding
- Trained by Meta AI Research

## 🚀 Features
- Side-by-side model comparison
- Real-time sentiment analysis
- Confidence scores
- Detailed interpretations
- Support for emojis and social media text

## 💻 Local Development

1. Clone the repository:
```bash
git clone https://github.com/dddmmm789/Social-Media-AI-Sentiment-Analysis-MVP.git
cd Social-Media-AI-Sentiment-Analysis-MVP
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run streamlit_app.py
```

## 📊 Training Data

### Simple Model Training Set
- 20 positive examples
- 20 neutral examples
- 20 negative examples

### RoBERTa Training
- Pre-trained on:
  - BookCorpus (11,038 books)
  - CC-News (63M news articles)
  - OpenWebText (Reddit content)
  - Stories (CommonCrawl data)
- Fine-tuned on millions of tweets

## 🔍 Key Differences
1. **Training Data Size**
   - Simple: 60 examples
   - RoBERTa: 160GB of text + millions of tweets

2. **Model Complexity**
   - Simple: Basic LSTM architecture
   - RoBERTa: Advanced transformer architecture

3. **Performance**
   - Simple: Good for basic sentiment
   - RoBERTa: Excellent for complex/nuanced sentiment

## 🤝 Contributing
Feel free to:
- Open issues
- Submit pull requests
- Suggest improvements
- Share feedback

## 📝 License
MIT License - feel free to use this code for your own projects!

## 🙏 Acknowledgments
- Meta AI Research for RoBERTa
- Stanford NLP for GloVe embeddings
- Streamlit for the web framework 