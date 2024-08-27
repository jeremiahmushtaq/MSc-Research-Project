import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
from matplotlib import pyplot as plt
import pandas as pd
from functions import load_haplotypes

"""
Setup and Loading
"""
# Set directory paths
high_dir = '/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Simulation Results/CNN Training Data/High'
medium_dir = '/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Simulation Results/CNN Training Data/Medium'
low_dir = '/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Simulation Results/CNN Training Data/Low'

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
min_hap_len = min(min_hap_len_high, min_hap_len_medium, min_hap_len_low)

# Define function to truncate training sequences
def truncate(category):
    trunc_seqs = []
    for sim in category:
        temp = [] 
        for ind_seq in sim:
            x = ind_seq[:min_hap_len]
            temp.append(x)
        trunc_seqs.append(temp)
    return trunc_seqs

# Truncate sequences
high_hap_trunc = np.array(truncate(high))
med_hap_trunc = np.array(truncate(medium))
low_hap_trunc = np.array(truncate(low))

# Build original dataset
original_data = np.concatenate((high_hap_trunc, med_hap_trunc, low_hap_trunc), axis=0)

"""
Data Augmentation
"""
# Flip horizontally along the width axis
flipped_data_horizontal = np.flip(original_data, axis=2)

# Flip vertically along the height axis
flipped_data_vertical = np.flip(original_data, axis=1)

# Flip along the depth axis
flipped_data_depth = np.flip(original_data, axis=0)

# Compile orginial and augmented datasets
data = np.concatenate((original_data, flipped_data_horizontal, flipped_data_vertical, flipped_data_depth), axis=0)

"""
Build Model
"""
# Define tensors
X = tf.constant(data)
X = tf.reshape(X, [X.shape[0], X.shape[1], X.shape[2], 1]) # (no. of simulations (batch size), no. of individuals (height), sequence length (width), no. of channels (amount of variables in sequence))

# Define labels
y = tf.constant(
    # Original
    [0]*100 + # High
    [1]*100 + # Medium
    [2]*100 + # Low
    # Horizontal
    [0]*100 + # High 
    [1]*100 + # Medium
    [2]*100 + # Low
    # Vertical
    [0]*100 + # High
    [1]*100 + # Medium
    [2]*100 + # Low
    # Depth
    [2]*100 + # High
    [1]*100 + # Medium
    [0]*100   # Low
)

# Define cnn layers
model = Sequential([
    Input(shape=(100, min_hap_len, 1)),
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
hist = model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)

# Save model
model.save('cnn.h5')

"""
Save Tabular Training Metrics
"""
# File logistics
output_directory = '/Users/jeremiahmushtaq/Documents/University/MSc Research Project/CNN'
output_file_name = 'metrics.tsv'
metrics_file_path = os.path.join(output_directory, output_file_name)

# Create dataframe with metrics
df = pd.DataFrame({
    'Epoch (%)': range(1, len(hist.history['accuracy']) + 1),
    'Training Accuracy (%)': hist.history['accuracy'],
    'Validation Accuracy (%)': hist.history['val_accuracy'],
    'Training Loss (%)': hist.history['loss'],
    'Validation Loss (%)': hist.history['val_loss']
})

# Format values
df['Training Accuracy (%)'] = df['Training Accuracy (%)'].apply(lambda x: f"{x * 100:.2f}")
df['Validation Accuracy (%)'] = df['Validation Accuracy (%)'].apply(lambda x: f"{x * 100:.2f}")
df['Training Loss (%)'] = df['Training Loss (%)'].apply(lambda x: f"{x:.2f}")
df['Validation Loss (%)'] = df['Validation Loss (%)'].apply(lambda x: f"{x:.2f}")

# Save the dataframe to a tsv file
df.to_csv(metrics_file_path, sep='\t', index=False)

"""
Save Graphical Training Metrics
"""
# Define the directory and file name
output_directory = '/Users/jeremiahmushtaq/Documents/University/MSc Research Project/CNN'
output_file_name = 'plot.png'
metrics_file_path = os.path.join(output_directory, output_file_name)

plt.subplot(1, 2, 1)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()

# Save the plot
plt.savefig(metrics_file_path)
print("Results saved as", output_file_name)

# User status update
print("Graphical metrics saved.")