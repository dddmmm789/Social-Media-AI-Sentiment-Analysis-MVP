import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from tqdm import tqdm

class SentimentModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_classes=3):
        super(SentimentModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim // 2)
    
    def attention_net(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.transpose(1, 2), lstm_output)
        return context.squeeze(1)
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        lstm_out, _ = self.lstm(embedded)
        attn_out = self.attention_net(lstm_out)
        
        out = self.dropout(attn_out)
        out = self.fc1(out)
        out = self.batch_norm1(out)
        out = F.relu(out)
        
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = F.relu(out)
        
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out

class SentimentAnalyzer:
    def __init__(self, model_path=None, use_pretrained=True):
        # Initialize parameters
        self.embedding_dim = 100
        self.hidden_dim = 256
        self.vocab_size = 5000  # Reduced vocabulary size
        self.max_length = 50
        
        # Initialize vocabulary with special tokens
        self.word_to_idx = {'<pad>': 0, '<unk>': 1}
        
        # Load basic vocabulary
        self.initialize_vocab()
        
        # Initialize model
        self.model = SentimentModel(
            self.embedding_dim,
            self.hidden_dim,
            self.vocab_size
        )
        
        # Load pre-trained model if provided and use_pretrained is True
        if model_path and use_pretrained:
            self.model.load_state_dict(torch.load(model_path))
        
        self.model.eval()
        
        # Set mode
        self.mode = "trained" if (model_path and use_pretrained) else "untrained"
    
    def initialize_vocab(self):
        # Initialize with some common words
        common_words = [
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'good', 'great', 'bad', 'terrible', 'awesome', 'amazing', 'poor',
            'excellent', 'horrible', 'wonderful', 'awful', 'perfect', 'worst',
            'best', 'love', 'hate', 'like', 'dislike', 'happy', 'sad'
        ]
        
        idx = len(self.word_to_idx)
        for word in common_words:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = idx
                idx += 1
    
    def preprocess_text(self, text):
        # Improved tokenization
        text = text.lower()
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b|[!?.,]', text)
        
        # Convert to indices
        indices = [self.word_to_idx.get(token, self.word_to_idx['<unk>']) for token in tokens]
        
        # Pad or truncate to max_length
        if len(indices) < self.max_length:
            indices += [self.word_to_idx['<pad>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        return torch.tensor(indices).unsqueeze(0)
    
    def analyze(self, text):
        """Analyze the sentiment of a given text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            tuple: (sentiment_label, sentiment_score, probabilities)
                sentiment_label is one of: 'positive', 'negative', 'neutral'
                sentiment_score is a float between -1 and 1
                probabilities is a list of three probabilities [neg, neu, pos]
        """
        # Preprocess text
        input_tensor = self.preprocess_text(text)
        
        # Get model prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            
            # Get predicted class
            predicted_class = torch.argmax(probabilities).item()
            
            # Convert to sentiment label
            sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            sentiment = sentiment_map[predicted_class]
            
            # Calculate sentiment score (-1 to 1)
            probs = probabilities.squeeze().tolist()
            score = probs[2] - probs[0]  # positive_prob - negative_prob
            
        return sentiment, score, probs

    def train(self, train_data, epochs=5, batch_size=32, learning_rate=0.001):
        """Train the sentiment analysis model.
        
        Args:
            train_data: List of (text, label) tuples
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        # Convert data to batches
        num_batches = (len(train_data) + batch_size - 1) // batch_size
        best_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(range(num_batches), desc=f'Epoch {epoch+1}')
            
            # Shuffle data
            np.random.shuffle(train_data)
            
            for i in progress_bar:
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(train_data))
                batch = train_data[start_idx:end_idx]
                texts, labels = zip(*batch)
                
                # Prepare batch
                input_tensors = torch.stack([self.preprocess_text(text).squeeze() for text in texts])
                label_tensor = torch.tensor(labels)
                
                # Forward pass
                outputs = self.model(input_tensors)
                loss = criterion(outputs, label_tensor)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                avg_loss = total_loss / (i + 1)
                progress_bar.set_description(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
            
            # Update learning rate
            avg_loss = total_loss / num_batches
            scheduler.step(avg_loss)
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), 'best_model.pt')
            
            print(f'Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}')
    
    def save_model(self, path):
        """Save the model to a file."""
        torch.save(self.model.state_dict(), path) 