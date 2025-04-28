from IPython import get_ipython
from IPython.display import display

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Lambda  # Import Lambda here
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np



# these dimensions of each file
D_word = 100     # Original word embedding dimension
D_char = 24      # Original character embedding dimension
D_stats = 4      # Dimension for statistical features
D_proj = 128     # Projection dimension for all modalities
n_class = 2      # Number of classes for classification

# Input layers
input_word  = Input(shape=(D_word,), name="word_embedding")
input_char  = Input(shape=(D_char,), name="char_embedding")
input_stats = Input(shape=(D_stats,), name="stats_features")

# Projection layers for each modality
proj_word  = Dense(D_proj, activation='relu')(input_word)
proj_char  = Dense(D_proj, activation='relu')(input_char)
proj_stats = Dense(D_proj, activation='relu')(input_stats)

# Stack the projections: shape (batch_size, 3, D_proj)
stacked_modalities = Lambda(lambda x: tf.stack(x, axis=1))([proj_word, proj_char, proj_stats])


# Attention Fusion Layer (inline implementation)
def attention_fusion(x):
    # x is shape (batch, 3, D_proj)
    attn_scores = Dense(1, activation='relu')(x)       # shape (batch, 3, 1)
    attn_scores = tf.squeeze(attn_scores, axis=-1)       # shape (batch, 3)
    attn_weights = tf.nn.softmax(attn_scores, axis=1)     # shape (batch, 3)
    attn_weights = tf.expand_dims(attn_weights, axis=-1)   # shape (batch, 3, 1)
    # Weighted sum
    fusion_out = tf.reduce_sum(x * attn_weights, axis=1)   # shape (batch, D_proj)
    return fusion_out

fusion_representation = tf.keras.layers.Lambda(attention_fusion, name="attention_fusion")(stacked_modalities)

# Further processing and classification
fusion_hidden = Dense(64, activation='relu')(fusion_representation)
fusion_hidden = Dropout(0.2)(fusion_hidden)
output = Dense(n_class, activation='softmax')(fusion_hidden)

# Build the model
model = Model(
    inputs=[input_word, input_char, input_stats],
    outputs=output,
    name="Attention_Fusion_Classifier"
)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Generate synthetic data for demonstration
num_samples = 1000

# Word embeddings: Class 0 has lower values, Class 1 has higher values
word_data_class0 = np.random.rand(num_samples // 2, D_word) * 0.4  # Class 0
word_data_class1 = np.random.rand(num_samples // 2, D_word) * 0.4 + 0.6  # Class 1
word_data = np.vstack([word_data_class0, word_data_class1])

# Character embeddings: Class 0 has lower values, Class 1 has higher values
char_data_class0 = np.random.rand(num_samples // 2, D_char) * 0.4  # Class 0
char_data_class1 = np.random.rand(num_samples // 2, D_char) * 0.4 + 0.6  # Class 1
char_data = np.vstack([char_data_class0, char_data_class1])

# Statistical features: Class 0 has lower values, Class 1 has higher values
stat_data_class0 = np.random.rand(num_samples // 2, D_stats) * 0.4  # Class 0
stat_data_class1 = np.random.rand(num_samples // 2, D_stats) * 0.4 + 0.6  # Class 1
stat_data = np.vstack([stat_data_class0, stat_data_class1])

# Labels: Class 0 (0) and Class 1 (1)
labels = np.array([0] * (num_samples // 2) + [1] * (num_samples // 2))

# Shuffle the data
indices = np.arange(num_samples)
np.random.shuffle(indices)
word_data = word_data[indices]
char_data = char_data[indices]
stat_data = stat_data[indices]
labels = labels[indices]

# Split data into training and testing sets (70:30)
X_word_train, X_word_test, X_char_train, X_char_test, X_stat_train, X_stat_test, y_train, y_test = train_test_split(
    word_data, char_data, stat_data, labels, test_size=0.3, random_state=42
)

# Hyperparameters
epochs = 100  # Train for 100 epochs
batch_size = 32

# Train the model
history = model.fit(
    [X_word_train, X_char_train, X_stat_train],
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=([X_word_test, X_char_test, X_stat_test], y_test),
    verbose=1
)

# Print validation accuracy every 10 epochs
print("\nValidation Accuracy Over Epochs:")
for epoch in range(0, epochs, 10):
    print(f"Epoch {epoch + 1}: Validation Accuracy = {history.history['val_accuracy'][epoch]:.4f}")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate([X_word_test, X_char_test, X_stat_test], y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Find the best epoch and corresponding accuracy
best_epoch = np.argmax(history.history['val_accuracy']) + 1
best_accuracy = history.history['val_accuracy'][best_epoch - 1]

# Print the best result
print(f"\nBest Epoch: {best_epoch}")
print(f"Best Validation Accuracy: {best_accuracy:.4f}")
print(f"Hyperparameters: Epochs={epochs}, Batch Size={batch_size}, Projection Dim={D_proj}, Hidden Dim=64, Dropout=0.2")
