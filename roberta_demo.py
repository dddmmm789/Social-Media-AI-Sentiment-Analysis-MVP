from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

def initialize_model():
    print("Loading RoBERTa model (this may take a moment)...")
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print("Model loaded successfully!")
    return classifier

def analyze_sentiment(classifier, text):
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    
    # Convert label to more readable format
    if label == 'LABEL_0':
        sentiment = 'NEGATIVE'
    elif label == 'LABEL_1':
        sentiment = 'NEUTRAL'
    else:
        sentiment = 'POSITIVE'
    
    return sentiment, score

def main():
    print("\n=== RoBERTa Sentiment Analysis Demo ===")
    print("This demo uses a pre-trained RoBERTa model fine-tuned for sentiment analysis.")
    print("Enter text to analyze (Ctrl+C to exit)\n")
    
    try:
        classifier = initialize_model()
        
        while True:
            try:
                text = input("\nEnter text to analyze: ").strip()
                if not text:
                    print("Please enter some text to analyze.")
                    continue
                
                sentiment, confidence = analyze_sentiment(classifier, text)
                print(f"\nText: {text}")
                print(f"Sentiment: {sentiment}")
                print(f"Confidence: {confidence:.2%}")
                
            except Exception as e:
                print(f"Error analyzing text: {str(e)}")
                
    except KeyboardInterrupt:
        print("\nThank you for using the RoBERTa Sentiment Analysis Demo!")

if __name__ == "__main__":
    main() 