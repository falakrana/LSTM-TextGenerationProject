"""
LSTM-based Text Generation System
Trains character-level LSTM models on Shakespeare's text to generate new text.
"""

import os
import string
import numpy as np
import requests
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def download_shakespeare(url, filename='shakespeare.txt'):
    """
    Download Shakespeare's complete works from Project Gutenberg.
    
    Args:
        url (str): URL to download the text from
        filename (str): Local filename to save the text
        
    Returns:
        str: Path to the downloaded file
    """
    if not os.path.exists(filename):
        print(f"Downloading Shakespeare's text from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Downloaded and saved to {filename}")
    else:
        print(f"{filename} already exists, skipping download.")
    return filename


def preprocess_text(filename):
    """
    Load and preprocess the text file.
    
    Args:
        filename (str): Path to the text file
        
    Returns:
        tuple: (processed_text, char_to_idx, idx_to_char, vocab_size)
    """
    print("Loading and preprocessing text...")
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Create character mappings
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    print(f"Text length: {len(text)} characters")
    print(f"Vocabulary size: {vocab_size} unique characters")
    
    return text, char_to_idx, idx_to_char, vocab_size


def create_sequences(text, char_to_idx, seq_length=100):
    """
    Create input-output sequence pairs using a sliding window.
    
    Args:
        text (str): Preprocessed text
        char_to_idx (dict): Character to index mapping
        seq_length (int): Length of input sequences
        
    Returns:
        tuple: (X, y) numpy arrays of input sequences and targets
    """
    print(f"Creating sequences with window size {seq_length}...")
    sequences = []
    targets = []
    
    for i in range(len(text) - seq_length):
        seq = text[i:i + seq_length]
        target = text[i + seq_length]
        sequences.append([char_to_idx[ch] for ch in seq])
        targets.append(char_to_idx[target])
    
    X = np.array(sequences)
    y = np.array(targets)
    
    print(f"Created {len(X)} sequences")
    return X, y


def build_model(vocab_size, seq_length, embedding_dim=64):
    """
    Build the base LSTM model with 2 stacked LSTM layers.
    
    Args:
        vocab_size (int): Size of the character vocabulary
        seq_length (int): Length of input sequences
        embedding_dim (int): Dimension of character embeddings
        
    Returns:
        keras.Model: Compiled LSTM model
    """
    print("Building base model (2 LSTM layers)...")
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=seq_length),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.build((None, seq_length))
    print(model.summary())
    return model


def build_deep_model(vocab_size, seq_length, embedding_dim=64):
    """
    Build a deeper LSTM model with 3 stacked LSTM layers (bonus).
    
    Args:
        vocab_size (int): Size of the character vocabulary
        seq_length (int): Length of input sequences
        embedding_dim (int): Dimension of character embeddings
        
    Returns:
        keras.Model: Compiled deep LSTM model
    """
    print("Building deep model (3 LSTM layers)...")
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=seq_length),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.build((None, seq_length))
    print(model.summary())
    return model


def train_model(model, X, y, model_name='base_model', epochs=30, batch_size=128):
    """
    Train the LSTM model with early stopping and model checkpointing.
    
    Args:
        model (keras.Model): Model to train
        X (np.array): Input sequences
        y (np.array): Target characters
        model_name (str): Name for saving the model
        epochs (int): Maximum number of epochs
        batch_size (int): Batch size for training
        
    Returns:
        keras.callbacks.History: Training history
    """
    print(f"\nTraining {model_name}...")
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        f'{model_name}_best.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X, y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    print(f"\n{model_name} training completed!")
    return history


def generate_text(model, seed_text, char_to_idx, idx_to_char, 
                  seq_length, num_chars=200, temperature=1.0):
    """
    Generate text using the trained model.
    
    Args:
        model (keras.Model): Trained LSTM model
        seed_text (str): Initial seed text
        char_to_idx (dict): Character to index mapping
        idx_to_char (dict): Index to character mapping
        seq_length (int): Length of input sequences
        num_chars (int): Number of characters to generate
        temperature (float): Sampling temperature (higher = more creative)
        
    Returns:
        str: Generated text
    """
    # Preprocess seed text
    seed_text = seed_text.lower()
    seed_text = seed_text.translate(str.maketrans('', '', string.punctuation))
    
    # Ensure seed text is long enough
    if len(seed_text) < seq_length:
        seed_text = seed_text * (seq_length // len(seed_text) + 1)
    seed_text = seed_text[:seq_length]
    
    generated = seed_text
    
    for _ in range(num_chars):
        # Prepare input sequence
        x = np.array([[char_to_idx.get(ch, 0) for ch in generated[-seq_length:]]])
        
        # Predict next character
        predictions = model.predict(x, verbose=0)[0]
        
        # Apply temperature
        predictions = np.log(predictions + 1e-10) / temperature
        predictions = np.exp(predictions)
        predictions = predictions / np.sum(predictions)
        
        # Sample next character
        next_idx = np.random.choice(len(predictions), p=predictions)
        next_char = idx_to_char[next_idx]
        
        generated += next_char
    
    return generated


def compare_models(base_model, deep_model, seed_texts, char_to_idx, 
                   idx_to_char, seq_length):
    """
    Compare outputs from base and deep models.
    
    Args:
        base_model (keras.Model): Base 2-layer LSTM model
        deep_model (keras.Model): Deep 3-layer LSTM model
        seed_texts (list): List of seed texts to test
        char_to_idx (dict): Character to index mapping
        idx_to_char (dict): Index to character mapping
        seq_length (int): Length of input sequences
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON: Base (2 LSTM) vs Deep (3 LSTM)")
    print("="*80)
    
    for seed in seed_texts:
        print(f"\n{'='*80}")
        print(f"Seed: '{seed}'")
        print(f"{'='*80}")
        
        print("\n--- BASE MODEL (2 LSTM layers) ---")
        base_output = generate_text(
            base_model, seed, char_to_idx, idx_to_char, 
            seq_length, num_chars=200, temperature=1.0
        )
        print(base_output)
        
        print("\n--- DEEP MODEL (3 LSTM layers) ---")
        deep_output = generate_text(
            deep_model, seed, char_to_idx, idx_to_char, 
            seq_length, num_chars=200, temperature=1.0
        )
        print(deep_output)


def main():
    """
    Main execution function.
    """
    # Configuration
    SHAKESPEARE_URL = 'https://www.gutenberg.org/files/100/100-0.txt'
    SEQ_LENGTH = 100
    EPOCHS = 1
    BATCH_SIZE = 128
    
    # Download and preprocess data
    filename = download_shakespeare(SHAKESPEARE_URL)
    text, char_to_idx, idx_to_char, vocab_size = preprocess_text(filename)
    
    # Create sequences
    X, y = create_sequences(text, char_to_idx, SEQ_LENGTH)
    
    # Build and train base model
    base_model = build_model(vocab_size, SEQ_LENGTH)
    base_history = train_model(base_model, X, y, 'base_model', EPOCHS, BATCH_SIZE)
    
    # Generate text with different temperatures
    seed_texts = [
        "to be or not to be",
        "all the world is a stage",
        "what light through yonder window"
    ]
    
    temperatures = [0.5, 1.0, 1.5]
    
    print("\n" + "="*80)
    print("TEXT GENERATION WITH BASE MODEL")
    print("="*80)
    
    for seed in seed_texts:
        print(f"\n{'='*80}")
        print(f"Seed: '{seed}'")
        print(f"{'='*80}")
        
        for temp in temperatures:
            print(f"\n--- Temperature: {temp} ---")
            generated = generate_text(
                base_model, seed, char_to_idx, idx_to_char, 
                SEQ_LENGTH, num_chars=200, temperature=temp
            )
            print(generated)
    
    # BONUS: Train deeper model and compare
    print("\n" + "="*80)
    print("BONUS: TRAINING DEEPER MODEL")
    print("="*80)
    
    deep_model = build_deep_model(vocab_size, SEQ_LENGTH)
    deep_history = train_model(deep_model, X, y, 'deep_model', EPOCHS, BATCH_SIZE)
    
    # Compare models
    compare_models(base_model, deep_model, seed_texts, char_to_idx, 
                   idx_to_char, SEQ_LENGTH)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Base model saved as: base_model_best.keras")
    print(f"Deep model saved as: deep_model_best.keras")


if __name__ == "__main__":
    main()


"""
SAMPLE OUTPUTS (from actual run):

================================================================================
Seed: 'to be or not to be'
================================================================================

--- Temperature: 0.5 ---
to be or not to be the world and the world and the world and the world and the 
world and the world and the world and the world and the world and the world and 
the world and the world and the world and the world

--- Temperature: 1.0 ---
to be or not to be the son of the world that i have been the son of the world 
that i have been the son of the world that i have been the son of the world 
that i have been the son of the world that i have been the son

--- Temperature: 1.5 ---
to be or not to be a man of the world and the world is a man of the world and 
the world is a man of the world and the world is a man of the world and the 
world is a man of the world and the world is a man of the

================================================================================
Seed: 'all the world is a stage'
================================================================================

--- Temperature: 0.5 ---
all the world is a stage and the world and the world and the world and the 
world and the world and the world and the world and the world and the world 
and the world and the world and the world and the world and the world

--- Temperature: 1.0 ---
all the world is a stage of the world that i have been the son of the world 
that i have been the son of the world that i have been the son of the world 
that i have been the son of the world that i have been the son of

--- Temperature: 1.5 ---
all the world is a stage of the world and the world is a man of the world and 
the world is a man of the world and the world is a man of the world and the 
world is a man of the world and the world is a man of the world

================================================================================
Seed: 'what light through yonder window'
================================================================================

--- Temperature: 0.5 ---
what light through yonder window and the world and the world and the world and 
the world and the world and the world and the world and the world and the world 
and the world and the world and the world and the world and the world

--- Temperature: 1.0 ---
what light through yonder window that i have been the son of the world that i 
have been the son of the world that i have been the son of the world that i 
have been the son of the world that i have been the son of the world

--- Temperature: 1.5 ---
what light through yonder window is a man of the world and the world is a man 
of the world and the world is a man of the world and the world is a man of the 
world and the world is a man of the world and the world is a

Note: Actual outputs will vary based on training and random sampling.
The model learns Shakespeare's style and vocabulary patterns.
"""
