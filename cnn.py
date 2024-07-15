import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense # type: ignore

# Set directory to access data
cwd = os.getcwd()
confirm_cwd = print(cwd)
nwd_path = os.path.join(cwd, "Simulation Results")
nwd = os.chdir(nwd_path)
confirm_nwd = print(os.getcwd())

# Remove unnecessary files from nwd
files = os.listdir()
files.remove('.DS_Store')
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

# Data pre-processing: Padding
haplotype_lenghts= [] # Extracting number of haplotypes for each simulation
for i in range(0,len(haplotype_data)):
    x = len(haplotype_data[i][0])
    haplotype_lenghts.append(x)

haplotype_lenghts = (list(haplotype_lenghts))
max_haplotype_length = max(haplotype_lenghts) # Determining longest haplotype length
confirm_max_haplotype_length = print(max_haplotype_length)

haplotype_data_padded = [] # Add '-1' padding
for i in range(0,len(haplotype_lenghts)):
    pad_width_2d = max_haplotype_length - haplotype_lenghts[i]
    x = np.pad(array=haplotype_data[i], pad_width=((0,0), (0,pad_width_2d)), mode='constant', constant_values=-1)
    haplotype_data_padded.append(x)

# Convert padded_data to a TensorFlow tensor
X = tf.constant(haplotype_data_padded, dtype=tf.float32)
confirm_tensor_shape = print("Shape of X:", X.shape)

# Define the Sequential model with Input layer
model = tf.keras.Sequential([
    Conv1D(32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Example training (replace with your actual training data and labels)
cateogory1 = 0 # low
cateogory2 = 1 # medium
cateogory3 = 2 # high
cateogory1 = [cateogory1] * 100
cateogory2 = [cateogory2] * 100
cateogory3 = [cateogory3] * 100
y = np.array(cateogory1+cateogory2+cateogory3)
model.fit(X, y, epochs=10, batch_size=32)