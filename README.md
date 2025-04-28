# Attention Fusion Classifier

A deep learning model that **combines multi-modal features** (word embeddings, character embeddings, and statistical features) based on a **custom attention mechanism** for classification.

It showcases how to fuse different input types and apply an attention-based fusion layer to yield improved performance.

## Features

- It accomodates three input modalities: Word Embeddings, Character Embeddings, and Statistical Features.
- Attention Fusion Layer to learn the weightage of every modality dynamically.
- Validation tracking and best epoch selection.
- Light, easy-to-train model.

## Model Architecture

```
Input Layers
├── Word Embeddings → Dense → Projected Word Features
├── Char Embeddings → Dense → Projected Char Features
└── Stats Features → Dense → Projected Stats Features

Stack Modalities → Apply Attention Fusion → Fused Representation
Fused Representation → Dense → Dropout → Dense → Output
```

- **Attention Fusion Layer:**
- Calculates attention weights across modalities.
- Performs a weighted sum to create a fused representation.

## Dependencies

- Python ≥ 3.7
- TensorFlow ≥ 2.0
- scikit-learn
- NumPy
- IPython (optional, for display)

Install them by:

```bash
pip install tensorflow scikit-learn numpy ipython
```

## How to Run
Clone the repository:

```bash
git clone https://github.com/yourusername/Attention-fusion-classifier.git
cd Attention-fusion-classifier
```

Run the script:
```python
python train_attention_fusion.py
```

## Hyperparameters

| Hyperparameter | Value |
|---------------------------|--------------|
| Word Embedding Dim | 100 |
| Char Embedding Dim | 24 |
| Statistical Feature Dim | 4
| Projection Dim | 128
| Hidden Dim (Dense Layer) | 64
| Dropout Rate | 0.2 |
| Batch Size | 32 |
| Epochs | 100 |
| Classes | 2 (Binary) |

## Example Output
- Validation Accuracy is printed every 10 epochs.
- Best epoch and its relative validation accuracy are printed.
- Final Test Accuracy is also printed.

```
Epoch 91: Validation Accuracy = 0.9467

Test Accuracy: 0.9533

Best Epoch: 91
Best Validation Accuracy: 0.9467
Hyperparameters: Epochs=100, Batch Size=32, Projection Dim=128, Hidden Dim=64, Dropout=0.2
```

## Project Structure

```
Attention-fusion-classifier/
├── content
├── output
├── README.md
├── train_attention_fusion.py
└── requirements.txt
```
## Future Enhancements
- Add configurable hyperparameters via command-line arguments.
- Visualizing attention weights.
- Saving and loading model checkpoints.
