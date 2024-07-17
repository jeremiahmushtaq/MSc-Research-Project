import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from matplotlib import pyplot as plt

"""
Setup and Loading
"""
# Define function to load haploypes
def load_haplotypes(directory):
    # Set directory to access data
    os.chdir(directory)
    print(f"Current working directory: {os.getcwd()}")

    # Remove .DS_Store
    files = [f for f in os.listdir() if f != '.DS_Store']
    print(f"Number of files: {len(files)}")

    # Import haplotypes
    haplotypes = []
    for file in files:
        x = np.loadtxt(file, delimiter='\t', dtype=int)
        haplotypes.append(x)
    
    return haplotypes

# Set directories' path
high_dir = '/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Simulation Results/Rep 1/High'
medium_dir = '/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Simulation Results/Rep 1/Medium'
low_dir = '/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Simulation Results/Rep 1/Low'

# Load haplotypes
high = load_haplotypes(high_dir)
medium = load_haplotypes(medium_dir)
low = load_haplotypes(low_dir)

"""
Data Preprocessing
"""
# Determine minimum sequence lengths
min_hap_len_high = min(len(j) for i in high for j in i)
min_hap_len_medium = min(len(j) for i in medium for j in i)
min_hap_len_low = min(len(j) for i in low for j in i)
min_hap_len = min(min_hap_len_high, min_hap_len_medium, min_hap_len_high)
print(min_hap_len) # 4882

# Define function to truncate sequences
def truncate(category):
    test = []
    for sim in category:
        temp = [] 
        for ind_seq in sim:
            x = ind_seq[:min_hap_len]
            temp.append(x)
        test.append(temp)
    return test

# Truncate sequences
high_hap_trunc = np.array(truncate(high))
med_hap_trunc = np.array(truncate(medium))
low_hap_trunc = np.array(truncate(low))

# Build dataset
data = np.concatenate((high_hap_trunc, med_hap_trunc, low_hap_trunc), axis=0)

"""
Build Model
"""
# Define tensors
X = tf.constant(data)
X = tf.reshape(X, [X.shape[0], X.shape[1], X.shape[2], 1]) # (no. of simulations (batch size), no. of individuals (height), sequence length (width), no. of channels (amount of variables in sequence))

# Define labels
y = tf.constant(
    [0]*100 + # High
    [1]*100 + # Medium
    [2]*100   # Low
)

# Define cnn layers
model = Sequential([
    Input(shape=(100, 4882, 1)),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(3, 3)),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(3, 3)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit model
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

"""
Training & Validation Evaluation
"""
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