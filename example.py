from sentiment_analyzer import SentimentAnalyzer

def main():
    # Initialize the analyzer with the best trained model
    analyzer = SentimentAnalyzer(model_path='best_model.pt')
    
    # Example texts
    example_texts = [
        # Positive examples
        "I absolutely love this product! It's amazing!",
        "The customer service team was incredibly helpful",
        "This is exactly what I needed, perfect solution",
        
        # Neutral examples
        "The product works as expected, nothing special",
        "It's okay for basic use, but could be better",
        "Average performance, reasonable price",
        
        # Negative examples
        "This is the worst experience ever. Terrible service.",
        "Complete waste of money and time",
        "Very disappointed with the quality",
        
        # Mixed/Complex examples
        "Good features but expensive",
        "Great product when it works, but has some bugs",
        "The new update has both improvements and issues",
        
        # Social media style examples
        "OMG this is so awesome! 😍",
        "meh... not worth the hype tbh",
        "worst. purchase. ever. 😡",
    ]
    
    print("Analyzing example texts...\n")
    
    # Analyze each text
    for text in example_texts:
        sentiment, score = analyzer.analyze(text)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Score: {score:.2f}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main() 