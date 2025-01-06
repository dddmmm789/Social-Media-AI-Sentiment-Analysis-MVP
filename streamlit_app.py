import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sentiment_analyzer import SentimentAnalyzer
import torch

# Initialize both models
@st.cache_resource
def load_roberta():
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return classifier

@st.cache_resource
def load_simple_model():
    return SentimentAnalyzer()

def analyze_roberta(classifier, text):
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    
    # Convert label to more readable format
    sentiment_map = {
        'negative': 'NEGATIVE',
        'neutral': 'NEUTRAL',
        'positive': 'POSITIVE'
    }
    sentiment = sentiment_map.get(label, label.upper())
    
    return sentiment, score

# Add multiprocessing support
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    # Page setup
    st.set_page_config(page_title="Sentiment Analysis Comparison", layout="wide")
    st.title("Sentiment Analysis Model Comparison")

    # Model Comparison Introduction
    st.markdown("""
    This demo showcases two different approaches to sentiment analysis:
    1. A simple custom-trained model using GloVe + LSTM
    2. Meta's RoBERTa model (state-of-the-art transformer)

    Try both models with the same text to see how they compare!
    """)

    # Load models
    with st.spinner("Loading models... (this may take a minute)"):
        roberta = load_roberta()
        simple_model = load_simple_model()

    # Input section
    st.write("Enter text below to analyze its sentiment using both models:")
    text_input = st.text_area("Text to analyze:", height=100)

    if st.button("Analyze Sentiment"):
        if text_input.strip():
            # Create two columns for side-by-side comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Simple Model (GloVe + LSTM)")
                sentiment, score = simple_model.analyze(text_input)
                st.write(f"**Sentiment:** {sentiment.upper()}")
                st.write(f"**Score:** {score:.2f} (range: -1 to 1)")
                
                # Add interpretation
                if abs(score) < 0.2:
                    st.write("**Interpretation:** Mostly neutral content")
                elif score > 0:
                    strength = "Strongly" if score > 0.6 else "Moderately"
                    st.write(f"**Interpretation:** {strength} positive content")
                else:
                    strength = "Strongly" if score < -0.6 else "Moderately"
                    st.write(f"**Interpretation:** {strength} negative content")
            
            with col2:
                st.subheader("Meta's RoBERTa Model")
                roberta_sentiment, roberta_confidence = analyze_roberta(roberta, text_input)
                st.write(f"**Sentiment:** {roberta_sentiment}")
                st.write(f"**Confidence:** {roberta_confidence:.2%}")
                
                # Add interpretation
                if roberta_confidence > 0.9:
                    st.write("**Interpretation:** Very high confidence in this prediction")
                elif roberta_confidence > 0.7:
                    st.write("**Interpretation:** Good confidence in this prediction")
                else:
                    st.write("**Interpretation:** Mixed or ambiguous sentiment")
        else:
            st.warning("Please enter some text to analyze.")

    # Detailed Model Comparison
    st.markdown("""
    ---
    ## Understanding the Models

    ### Simple Model (GloVe + LSTM)
    #### Base Components:
    - **GloVe Word Embeddings**: Pre-trained on 6 billion tokens from Wikipedia + Gigaword
    - **Architecture**: Bidirectional LSTM with attention mechanism
    - **Output Layer**: 3-way classification (Positive, Neutral, Negative)

    #### Training Data (60 examples):
    **Positive Examples (20):**
    ```python
    "I absolutely love this product! It's amazing!"
    "This is amazing, best experience ever!"
    "Great service and fantastic quality"
    "Very happy with my purchase"
    "Excellent work, highly recommended"
    # ... and 15 more similar examples
    ```

    **Neutral Examples (20):**
    ```python
    "It's okay, nothing special"
    "The product works as expected"
    "Average experience, could be better"
    "Not bad, not great"
    "Standard service, nothing extraordinary"
    # ... and 15 more similar examples
    ```

    **Negative Examples (20):**
    ```python
    "This is terrible, worst experience ever"
    "Very disappointed with the service"
    "Poor quality, would not recommend"
    "Waste of money, don't buy this"
    "Horrible customer service"
    # ... and 15 more similar examples
    ```

    ### Meta's RoBERTa Model
    #### Base Components:
    - **Architecture**: Robustly optimized BERT approach (by Meta AI)
    - **Innovation**: Improved training methodology over BERT
    - **Size**: 355M parameters in base model
    - **Training Compute**: 1024 V100 GPUs

    #### Training Data:
    - **Pre-training Corpus**: 160GB of text including:
      - BookCorpus (11,038 unpublished books)
      - CC-News (63 million English news articles)
      - OpenWebText (web content from Reddit)
      - Stories (a subset of CommonCrawl data)
    - **Fine-tuning**: Twitter sentiment dataset with millions of tweets
    - **Validation**: Extensive testing on social media content

    ## Key Differences

    ### Training Approach
    - **Simple Model**: Trained from scratch on 60 carefully selected examples
    - **RoBERTa**: Pre-trained by Meta AI on 160GB of text, fine-tuned on millions of social media posts

    ### Model Size
    - **Simple Model**: ~100MB (mostly GloVe embeddings)
    - **RoBERTa**: ~355MB (full model with optimized architecture)

    ### Capabilities
    - **Simple Model**: Good at basic sentiment patterns it was trained on
    - **RoBERTa**: State-of-the-art understanding of:
      - Context and nuance
      - Social media language
      - Emojis and informal text
      - Complex emotional expressions

    ### Speed
    - **Simple Model**: Faster inference (simpler architecture)
    - **RoBERTa**: More computational overhead but significantly better accuracy

    ### Why RoBERTa?
    - Developed by Meta AI Research
    - Improved training methodology over BERT
    - Specifically optimized for social media content
    - State-of-the-art results on sentiment analysis
    - Better handling of informal language and emojis

    ---
    ### Tips for Testing
    - Try simple phrases like "I love this!" or "This is terrible"
    - Try complex sentences with mixed sentiment
    - Try sentences about serious topics
    - Try sentences with emojis or slang
    """)

    # Add GitHub link
    st.markdown("""
    ---
    ### Want to Learn More?
    Check out the [GitHub repository](https://github.com/yourusername/Social-Media-AI-Sentiment-Analysis-MVP) for the complete code and documentation.
    """) 