import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Reshape # type: ignore
import matplotlib.pyplot as plt

# Set directory to access data
cwd = os.getcwd()
confirm_cwd = print(cwd)
nwd_path = os.path.join(cwd, "Simulation Results")
nwd = os.chdir(nwd_path)
confirm_nwd = print(os.getcwd())

# Remove unnecessary files from nwd
files = os.listdir()
files.remove('.DS_Store')
files.sort()
confirm_file_numbers = print(len(files))

# Import haplotypes
haplotype_data = []
for i in files:
    x = np.loadtxt(i, delimiter='\t', dtype=int)
    haplotype_data.append(x)

confirm_data_correctly_imported = print(
    "Number of simulations:", len(haplotype_data),
    '\n',
    "Number of individuals:", len(haplotype_data[0]),
    '\n',
    "Number of SNPs:", len(haplotype_data[0][0])
)

# Data pre-processing: Truncating
min_haplotype_length = min(len(j) for i in haplotype_data for j in i)
confirm_min_length = print("Shortest Sequence Length:", min_haplotype_length)
haplotype_data_truncated = [[j[:min_haplotype_length] for j in i] for i in haplotype_data]
confirm_min_trun = print("Truncated Sequence Length:", len(haplotype_data_truncated[0][0]))

# Convert truncated to a TensorFlow tensor
haplotype_data_truncated = np.array(haplotype_data_truncated)
X = tf.constant(haplotype_data_truncated)
X = tf.reshape(X, [(X.shape[0]*X.shape[1]), X.shape[2], 1, 1]) # (batch_size, length, channels)

# Define Labels
y = tf.constant(
    [0]*10000 + # Meidum
    [1]*10000 + # High
    [2]*10000   # Low
)
y = tf.repeat(y, repeats=100)

# Define CNN
model = Sequential([
    Input(shape=(min_haplotype_length, 1, 1)), # (None, 4882, 1, 1)
    Conv2D(filters=64, kernel_size=(3, 1), activation='relu'), # (None, 4880, 1, 64) 
    MaxPooling2D(pool_size=(2, 1)), # (None, 2440, 1, 64)
    Conv2D(filters=32, kernel_size=(3, 1), activation='relu'), # (None, 2438, 1, 32)
    MaxPooling2D(pool_size=(2, 1)), # (None, 1219, 1, 32)
    Flatten(), # (None, 39008)
    Dense(64, activation='relu'), # (None, 64)
    Dense(3, activation='softmax') # (None, 3)
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with your reshaped data and expanded labels
history = model.fit(X, y, epochs=10, batch_size=128, validation_split=0.2)

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.show()