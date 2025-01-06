from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

class AdvancedSentimentAnalyzer:
    def __init__(self):
        print("Loading pre-trained model from Meta/HuggingFace...")
        # Load RoBERTa model fine-tuned for sentiment analysis
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True
        )
        print("Model loaded successfully!")

    def analyze(self, text):
        # Get sentiment scores
        results = self.sentiment_analyzer(text)[0]
        
        # Convert scores to our format
        scores = {item['label']: item['score'] for item in results}
        
        # Determine sentiment
        max_sentiment = max(results, key=lambda x: x['score'])
        
        # Calculate normalized score (-1 to 1)
        positive_score = scores.get('positive', 0)
        negative_score = scores.get('negative', 0)
        neutral_score = scores.get('neutral', 0)
        
        # Calculate final score
        if max_sentiment['label'] == 'neutral':
            final_score = 0
        else:
            final_score = positive_score - negative_score
        
        return {
            'sentiment': max_sentiment['label'],
            'score': final_score,
            'detailed_scores': {
                'positive': positive_score,
                'negative': negative_score,
                'neutral': neutral_score
            }
        }

def main():
    print("\n=== Advanced Sentiment Analyzer (Using Meta's RoBERTa) ===")
    print("This version uses a sophisticated pre-trained model.")
    print("Press Ctrl+C to exit.\n")
    
    # Initialize the analyzer
    analyzer = AdvancedSentimentAnalyzer()
    
    try:
        while True:
            # Get user input
            text = input("\nEnter text to analyze: ")
            
            if not text.strip():
                print("Please enter some text!")
                continue
            
            # Analyze sentiment
            result = analyzer.analyze(text)
            
            # Print results
            print("\nResults:")
            print("-" * 50)
            print(f"Text: {text}")
            print(f"Primary Sentiment: {result['sentiment']}")
            print(f"Overall Score: {result['score']:.2f}")
            print("\nDetailed Scores:")
            print(f"→ Positive: {result['detailed_scores']['positive']:.3f}")
            print(f"→ Neutral:  {result['detailed_scores']['neutral']:.3f}")
            print(f"→ Negative: {result['detailed_scores']['negative']:.3f}")
            
            # Print interpretation
            print("\nInterpretation:")
            if abs(result['score']) < 0.2:
                print("→ Mostly neutral content")
            elif result['score'] > 0:
                strength = "Strongly" if result['score'] > 0.6 else "Moderately"
                print(f"→ {strength} positive content")
            else:
                strength = "Strongly" if result['score'] < -0.6 else "Moderately"
                print(f"→ {strength} negative content")
            
            print("\n" + "-" * 50)
            
    except KeyboardInterrupt:
        print("\n\nThank you for using the Advanced Sentiment Analyzer!")
        print("Goodbye!\n")

if __name__ == "__main__":
    main() 