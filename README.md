# LSTM Text Generation with Shakespeare

A character-level LSTM neural network that learns to generate text in the style of William Shakespeare. This project implements both a base 2-layer LSTM model and a deeper 3-layer LSTM model, comparing their performance on text generation tasks.

## Overview

This project trains recurrent neural networks (LSTMs) on Shakespeare's complete works to learn patterns in language and generate new text. The model learns character-by-character sequences and can generate coherent text based on seed phrases.

## Features

- **Automatic Data Download**: Fetches Shakespeare's complete works from Project Gutenberg
- **Character-Level Learning**: Trains on individual characters for fine-grained text generation
- **Multiple Model Architectures**: 
  - Base model with 2 stacked LSTM layers
  - Deep model with 3 stacked LSTM layers
- **Temperature-Based Sampling**: Control creativity vs. coherence with temperature parameter
- **Model Comparison**: Side-by-side comparison of base and deep model outputs
- **Training Callbacks**: Early stopping and model checkpointing for optimal results

## Requirements

- Python 3.8+
- TensorFlow 2.13.0+
- NumPy 1.24.0+
- Requests 2.31.0+

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv lstmEnv
source lstmEnv/Scripts/activate  # On Windows
# source lstmEnv/bin/activate    # On Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main script to train both models and generate text:

```bash
python lstm_text_gen.py
```

This will:
1. Download Shakespeare's complete works (if not already present)
2. Preprocess the text and create character mappings
3. Train the base 2-layer LSTM model
4. Generate text samples with different temperature settings
5. Train the deeper 3-layer LSTM model
6. Compare outputs from both models

### Generated Files

After training, the following files will be created:
- `shakespeare.txt` - Downloaded Shakespeare text corpus
- `base_model_best.keras` - Best base model checkpoint
- `deep_model_best.keras` - Best deep model checkpoint

## Model Architecture

### Base Model (2 LSTM Layers)
```
- Embedding Layer (64 dimensions)
- LSTM Layer (128 units, return sequences)
- Dropout (0.2)
- LSTM Layer (64 units)
- Dropout (0.2)
- Dense Layer (vocab_size, softmax activation)
```

### Deep Model (3 LSTM Layers)
```
- Embedding Layer (64 dimensions)
- LSTM Layer (128 units, return sequences)
- Dropout (0.2)
- LSTM Layer (128 units, return sequences)
- Dropout (0.2)
- LSTM Layer (64 units)
- Dropout (0.2)
- Dense Layer (vocab_size, softmax activation)
```

## Configuration

Key parameters can be modified in the `main()` function:

```python
SEQ_LENGTH = 100      # Length of input sequences
EPOCHS = 30           # Maximum training epochs
BATCH_SIZE = 128      # Training batch size
```

## Temperature Parameter

The temperature parameter controls the randomness of text generation:

- **Low temperature (0.5)**: More conservative, repetitive, coherent text
- **Medium temperature (1.0)**: Balanced creativity and coherence
- **High temperature (1.5)**: More creative, diverse, potentially less coherent text

## Example Output

```
Seed: "to be or not to be"

Temperature: 0.5
to be or not to be the world and the world and the world...

Temperature: 1.0
to be or not to be the son of the world that i have been...

Temperature: 1.5
to be or not to be a man of the world and the world is a man...
```

## Training Details

- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam
- **Validation Split**: 20% of training data
- **Early Stopping**: Monitors validation loss with patience of 3 epochs
- **Model Checkpointing**: Saves best model based on validation loss

## Project Structure

```
.
├── lstm_text_gen.py      # Main script with all functionality
├── requirements.txt      # Python dependencies
├── shakespeare.txt       # Downloaded training corpus
├── DemoText.txt         # Sample Shakespeare text
├── lstmEnv/             # Virtual environment (if created)
├── base_model_best.keras    # Saved base model
└── deep_model_best.keras    # Saved deep model
```

## Functions

- `download_shakespeare()` - Downloads training data from Project Gutenberg
- `preprocess_text()` - Cleans and prepares text, creates character mappings
- `create_sequences()` - Generates training sequences using sliding window
- `build_model()` - Constructs base 2-layer LSTM architecture
- `build_deep_model()` - Constructs deeper 3-layer LSTM architecture
- `train_model()` - Trains model with callbacks
- `generate_text()` - Generates new text from trained model
- `compare_models()` - Compares outputs from different models

## Performance Notes

- Training time depends on hardware (GPU recommended for faster training)
- The model learns character-level patterns, not word-level semantics
- Longer training (more epochs) generally improves coherence
- The deep model may capture more complex patterns but requires more training time

## Limitations

- Character-level generation can produce non-words
- Model may memorize common phrases from training data
- Requires significant training data for good results
- Generated text may not always be grammatically correct

## Future Improvements

- Implement word-level LSTM for better semantic understanding
- Add attention mechanisms for improved context handling
- Experiment with different architectures (GRU, Transformer)
- Fine-tune hyperparameters for better performance
- Add text quality metrics for objective evaluation

## License

This project uses Shakespeare's works from Project Gutenberg, which are in the public domain.

## Acknowledgments

- Training data from [Project Gutenberg](https://www.gutenberg.org/)
- Built with TensorFlow/Keras
- Inspired by character-level language modeling research
