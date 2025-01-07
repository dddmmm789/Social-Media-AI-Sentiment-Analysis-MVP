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
st.title("🤖 Sentiment Analysis Journey: From Simple to Advanced")
st.markdown("""
### Experience the Evolution of Sentiment Analysis Models

This demo showcases three different stages of sentiment analysis models:
1. **Untrained Simple Model**: Raw GloVe + LSTM without training
2. **Trained Simple Model**: Same architecture trained on our dataset
3. **Advanced Model (RoBERTa)**: Meta's powerful transformer model
""")

# Model descriptions
st.header("📚 Model Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Untrained Model")
    st.markdown("""
    **Architecture:**
    - GloVe word embeddings (100d)
    - Bidirectional LSTM
    - Random weights
    - No training data
    
    **Purpose:**
    See how a model performs with just
    pre-trained word embeddings but
    no task-specific training
    """)

with col2:
    st.subheader("Trained Simple Model")
    st.markdown("""
    **Training Data:**
    - 60 manually curated examples
    - 20 positive, 20 neutral, 20 negative
    - Focus on basic sentiment patterns
    
    **Architecture:**
    - Same as untrained model
    - Weights optimized for sentiment
    - Model size: ~5MB
    """)

with col3:
    st.subheader("Advanced Model (RoBERTa)")
    st.markdown("""
    **Training Data:**
    - Pre-trained on 58M tweets
    - Fine-tuned on sentiment data
    - Extensive vocabulary
    
    **Architecture:**
    - 12-layer transformer
    - Self-attention mechanism
    - 768 hidden dimensions
    - Model size: ~500MB
    """)

@st.cache_resource
def load_models():
    # Load untrained model
    untrained_model = SentimentAnalyzer(use_pretrained=False)
    
    # Load trained model
    trained_model = SentimentAnalyzer(use_pretrained=True)
    
    # Load RoBERTa
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    return untrained_model, trained_model, (roberta_model, tokenizer)

# Load models
try:
    untrained_model, trained_model, (roberta_model, roberta_tokenizer) = load_models()
    
    # Input text
    st.header("✍️ Enter Text to Analyze")
    text_input = st.text_area("Type or paste text here:", height=100)
    
    if st.button("Analyze Sentiment"):
        if text_input:
            # Display results
            st.header("📊 Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Untrained Model")
                untrained_label, untrained_score, untrained_probs = untrained_model.analyze(text_input)
                st.write(f"**Sentiment:** {untrained_label}")
                st.write(f"**Score:** {untrained_score:.2f}")
                
                # Show probabilities
                st.write("**Probabilities:**")
                labels = ["Negative", "Neutral", "Positive"]
                for label, prob in zip(labels, untrained_probs):
                    st.write(f"{label}: {prob:.2%}")
            
            with col2:
                st.subheader("Trained Model")
                trained_label, trained_score, trained_probs = trained_model.analyze(text_input)
                st.write(f"**Sentiment:** {trained_label}")
                st.write(f"**Score:** {trained_score:.2f}")
                
                # Show probabilities
                st.write("**Probabilities:**")
                for label, prob in zip(labels, trained_probs):
                    st.write(f"{label}: {prob:.2%}")
            
            with col3:
                st.subheader("RoBERTa Model")
                # Process with RoBERTa
                inputs = roberta_tokenizer(text_input, return_tensors="pt", truncation=True, max_length=512)
                outputs = roberta_model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                
                # Map predictions to labels
                pred_idx = torch.argmax(probabilities).item()
                confidence = probabilities[0][pred_idx].item()
                
                st.write(f"**Sentiment:** {labels[pred_idx]}")
                st.write(f"**Confidence:** {confidence:.2%}")
                
                # Show all probabilities
                st.write("**Probabilities:**")
                for label, prob in zip(labels, probabilities[0]):
                    st.write(f"{label}: {prob.item():.2%}")
        else:
            st.warning("Please enter some text to analyze.")

    # Add an educational section
    st.header("🎓 Learning from Model Comparison")
    st.markdown("""
    This demonstration helps understand:
    1. **Impact of Training**: Compare untrained vs trained models to see how training data shapes performance
    2. **Model Complexity**: See how different architectures affect predictions
    3. **Confidence Patterns**: Notice how models express uncertainty differently
    4. **Real-world Applications**: Understand when simpler models might be sufficient
    """)

    # Training Examples Section
    st.header("📚 Training Dataset")
    st.markdown("""
    Below are example sentences from our training dataset. These 60 carefully chosen examples 
    were used to train the Simple Model, transforming it from the Untrained to the Trained version.
    """)

    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        st.subheader("😊 Positive Examples")
        st.markdown("""
        1. "I absolutely love this product!"
        2. "This is the best experience ever!"
        3. "Great service and fantastic quality"
        4. "Very happy with my purchase"
        """)

    with example_col2:
        st.subheader("😐 Neutral Examples")
        st.markdown("""
        1. "It's okay, nothing special"
        2. "The product works as expected"
        3. "Average experience overall"
        4. "Not bad, not great either"
        """)

    with example_col3:
        st.subheader("😞 Negative Examples")
        st.markdown("""
        1. "This is terrible, worst experience"
        2. "Very disappointed with the service"
        3. "Poor quality, would not recommend"
        4. "Complete waste of money"
        """)
            
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.info("Please make sure all required dependencies are installed and try again.") 