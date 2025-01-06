from sentiment_analyzer import SentimentAnalyzer

def main():
    print("\n=== Social Media Sentiment Analyzer ===")
    print("Enter text to analyze its sentiment.")
    print("Press Ctrl+C to exit.\n")
    
    # Initialize the analyzer with the trained model
    analyzer = SentimentAnalyzer(model_path='best_model.pt')
    
    try:
        while True:
            # Get user input
            text = input("\nEnter text to analyze: ")
            
            if not text.strip():
                print("Please enter some text!")
                continue
            
            # Analyze sentiment
            sentiment, score = analyzer.analyze(text)
            
            # Print results with color coding
            print("\nResults:")
            print("-" * 40)
            print(f"Text: {text}")
            print(f"Sentiment: {sentiment}")
            print(f"Score: {score:.2f}")
            
            # Print interpretation
            print("\nInterpretation:")
            if score > 0.7:
                print("→ Very positive sentiment")
            elif score > 0.3:
                print("→ Moderately positive sentiment")
            elif score > -0.3:
                print("→ Neutral sentiment")
            elif score > -0.7:
                print("→ Moderately negative sentiment")
            else:
                print("→ Very negative sentiment")
            
            print("\n" + "-" * 40)
            
    except KeyboardInterrupt:
        print("\n\nThank you for using the Sentiment Analyzer!")
        print("Goodbye!\n")

if __name__ == "__main__":
    main() 