import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
from matplotlib import pyplot as plt
import pandas as pd

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
high_dir = '/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Simulation Results/Rep 1 (100)/High'
medium_dir = '/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Simulation Results/Rep 1 (100)/Medium'
low_dir = '/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Simulation Results/Rep 1 (100)/Low'

# Load haplotypes
high = load_haplotypes(high_dir)
medium = load_haplotypes(medium_dir)
low = load_haplotypes(low_dir)

# User status update
print("Completed Setup & Loading.")

"""
Data Preprocessing
"""
# Determine minimum sequence lengths
min_hap_len_high = min(len(j) for i in high for j in i)
min_hap_len_medium = min(len(j) for i in medium for j in i)
min_hap_len_low = min(len(j) for i in low for j in i)
min_hap_len = min(min_hap_len_high, min_hap_len_medium, min_hap_len_low)
print("Minimum Sequence Length:", min_hap_len)

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

# User status update
print("Completed Data Preprocessing.")

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
hist = model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)

# User status update
print("Designed and fitted model.")

"""
Save Tabular Metrics
"""
# File logistics
output_directory = '/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Evaluation Logs/Tables/Multilayer Testing'
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
print("Results saved as", output_file_name)

# User status update
print("Tabular metrics saved.")

"""
Save Graphical Metrics
"""
# Define the directory and file name
output_directory = '/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Evaluation Logs/Images/Multilayer Testing'
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