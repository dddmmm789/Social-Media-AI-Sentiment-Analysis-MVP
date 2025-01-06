import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentiment_analyzer import SentimentAnalyzer

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis Model Comparison",
    page_icon="🤖",
    layout="wide"
)

# Title and introduction
st.title("🤖 Sentiment Analysis Model Comparison")
st.markdown("""
### Compare a Simple LSTM Model vs Meta's RoBERTa Model

This demo showcases two different approaches to sentiment analysis:
1. **Simple Model (GloVe + LSTM)**: A lightweight model trained on a small dataset
2. **Advanced Model (RoBERTa)**: Meta's powerful transformer model fine-tuned for sentiment analysis
""")

# Model descriptions
st.header("📚 Model Details")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Simple Model (GloVe + LSTM)")
    st.markdown("""
    **Training Data:**
    - 60 manually curated examples
    - 20 positive, 20 neutral, 20 negative
    - Focus on basic sentiment patterns
    
    **Base Components:**
    - GloVe word embeddings (100d)
    - Bidirectional LSTM
    - 2 fully connected layers
    - Model size: ~5MB
    """)

with col2:
    st.subheader("Advanced Model (RoBERTa)")
    st.markdown("""
    **Training Data:**
    - Pre-trained on 58M tweets
    - Fine-tuned on Stanford Sentiment Treebank
    - Extensive vocabulary and context understanding
    
    **Base Components:**
    - 12-layer transformer architecture
    - Self-attention mechanism
    - 768 hidden dimensions
    - Model size: ~500MB
    """)

@st.cache_resource
def load_simple_model():
    model = SentimentAnalyzer()
    return model

@st.cache_resource
def load_roberta_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

# Load models
try:
    simple_model = load_simple_model()
    roberta_model, roberta_tokenizer = load_roberta_model()
    
    # Input text
    st.header("✍️ Enter Text to Analyze")
    text_input = st.text_area("Type or paste text here:", height=100)
    
    if st.button("Analyze Sentiment"):
        if text_input:
            # Display results
            st.header("📊 Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Simple Model Results")
                simple_label, simple_score = simple_model.analyze(text_input)
                st.write(f"**Sentiment:** {simple_label}")
                st.write(f"**Score:** {simple_score:.2f}")
                
                # Interpretation
                if simple_score >= 0.5:
                    confidence = "Very Positive"
                elif simple_score > 0.2:
                    confidence = "Moderately Positive"
                elif simple_score > -0.2:
                    confidence = "Neutral"
                elif simple_score > -0.5:
                    confidence = "Moderately Negative"
                else:
                    confidence = "Very Negative"
                st.write(f"**Interpretation:** {confidence}")
            
            with col2:
                st.subheader("RoBERTa Model Results")
                # Process with RoBERTa
                inputs = roberta_tokenizer(text_input, return_tensors="pt", truncation=True, max_length=512)
                outputs = roberta_model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                
                # Map predictions to labels
                labels = ["Negative", "Neutral", "Positive"]
                pred_idx = torch.argmax(probabilities).item()
                confidence = probabilities[0][pred_idx].item()
                
                st.write(f"**Sentiment:** {labels[pred_idx]}")
                st.write(f"**Confidence:** {confidence:.2%}")
                
                # Show all probabilities
                st.write("**Detailed Probabilities:**")
                for label, prob in zip(labels, probabilities[0]):
                    st.write(f"{label}: {prob.item():.2%}")
        else:
            st.warning("Please enter some text to analyze.")

    # Training Examples Section
    st.header("🎓 Training Dataset")
    st.markdown("""
    Below are the actual sentences used to train the Simple Model. This demonstrates how even a small, 
    carefully curated dataset of 60 sentences can be used to train a basic sentiment analyzer.
    """)

    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        st.subheader("😊 Positive Training Sentences")
        st.markdown("""
        1. "I absolutely love this product! It's amazing!"
        2. "This is the best experience ever!"
        3. "Great service and fantastic quality"
        4. "Very happy with my purchase"
        5. "Excellent work, highly recommended"
        6. "Outstanding performance and results"
        7. "Incredible value for money"
        8. "Exceeded all my expectations"
        """)

    with example_col2:
        st.subheader("😐 Neutral Training Sentences")
        st.markdown("""
        1. "It's okay, nothing special"
        2. "The product works as expected"
        3. "Average experience, could be better"
        4. "Not bad, not great"
        5. "Standard service, nothing extraordinary"
        6. "Meets basic requirements"
        7. "Reasonable price for what you get"
        8. "Neither impressed nor disappointed"
        """)

    with example_col3:
        st.subheader("😞 Negative Training Sentences")
        st.markdown("""
        1. "This is terrible, worst experience ever"
        2. "Very disappointed with the service"
        3. "Poor quality, would not recommend"
        4. "Waste of money, don't buy this"
        5. "Horrible customer service"
        6. "Complete failure to deliver"
        7. "Extremely frustrated with this"
        8. "Regret making this purchase"
        """)
            
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.info("Please make sure all required dependencies are installed and try again.") 