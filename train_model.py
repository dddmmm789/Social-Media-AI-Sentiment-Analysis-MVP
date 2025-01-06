from sentiment_analyzer import SentimentAnalyzer
import torch

def get_training_data():
    """Create a training dataset.
    Labels: 0 = negative, 1 = neutral, 2 = positive
    """
    return [
        # Positive examples
        ("I absolutely love this product! It's amazing!", 2),
        ("This is amazing, best experience ever!", 2),
        ("Great service and fantastic quality", 2),
        ("Very happy with my purchase", 2),
        ("Excellent work, highly recommended", 2),
        ("The customer service was outstanding!", 2),
        ("This exceeded all my expectations", 2),
        ("A perfect solution to my problem", 2),
        ("Incredibly satisfied with the results", 2),
        ("Best purchase I've made this year", 2),
        ("The team was incredibly helpful", 2),
        ("Such a wonderful experience", 2),
        ("This made my day so much better", 2),
        ("Absolutely brilliant service", 2),
        ("I'm impressed by the quality", 2),
        ("The product works flawlessly!", 2),
        ("Outstanding customer support", 2),
        ("This is exactly what I needed", 2),
        ("Couldn't be happier with my choice", 2),
        ("Five stars, would recommend!", 2),
        
        # Neutral examples
        ("It's okay, nothing special", 1),
        ("The product works as expected", 1),
        ("Average experience, could be better", 1),
        ("Not bad, not great", 1),
        ("Standard service, nothing extraordinary", 1),
        ("It serves its purpose", 1),
        ("Reasonable quality for the price", 1),
        ("Met basic expectations", 1),
        ("Neither good nor bad", 1),
        ("Could use some improvements", 1),
        ("Some good points, some bad", 1),
        ("Middle of the road experience", 1),
        ("Does the job, but nothing more", 1),
        ("Acceptable but not outstanding", 1),
        ("Fair value for money", 1),
        ("Just another average product", 1),
        ("Nothing to complain about", 1),
        ("It's fine for basic use", 1),
        ("Gets the job done", 1),
        ("Mediocre but usable", 1),
        
        # Negative examples
        ("This is terrible, worst experience ever", 0),
        ("Very disappointed with the service", 0),
        ("Poor quality, would not recommend", 0),
        ("Waste of money, don't buy this", 0),
        ("Horrible customer service", 0),
        ("Complete disaster from start to finish", 0),
        ("Absolutely frustrated with this product", 0),
        ("The worst service I've ever received", 0),
        ("Total waste of time and money", 0),
        ("Extremely dissatisfied customer", 0),
        ("Terrible experience, avoid at all costs", 0),
        ("This product is a complete joke", 0),
        ("Deeply regret this purchase", 0),
        ("Unacceptable quality and service", 0),
        ("Stay away from this product", 0),
        ("Completely useless product", 0),
        ("Awful experience overall", 0),
        ("Don't waste your time", 0),
        ("Extremely poor quality", 0),
        ("This is a scam", 0),
    ]

def main():
    # Initialize the analyzer
    analyzer = SentimentAnalyzer()
    
    # Get training data
    train_data = get_training_data()
    
    print("Training the model...")
    # Train the model with improved hyperparameters
    analyzer.train(
        train_data,
        epochs=50,
        batch_size=16,
        learning_rate=0.001
    )
    
    # Load the best model (saved during training)
    analyzer.model.load_state_dict(torch.load('best_model.pt'))
    
    # Test the model with some examples
    test_texts = [
        "I really enjoy using this application",
        "This is the worst product I've ever used",
        "The service was acceptable",
        "Not sure how I feel about this",
        "The quality is outstanding!",
        "This needs significant improvements",
        "A decent product with room for improvement",
        "Absolutely terrible customer support",
        "Just what I was looking for!",
        "It's alright, but a bit expensive",
        "This product changed my life!",
        "I regret buying this",
        "Pretty standard features",
        "Amazing value for money",
        "Complete waste of resources",
    ]
    
    print("\nTesting the trained model:")
    for text in test_texts:
        sentiment, score = analyzer.analyze(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Score: {score:.2f}")

if __name__ == "__main__":
    main() 