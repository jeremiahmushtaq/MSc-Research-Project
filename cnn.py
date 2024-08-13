import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras.models import load_model  # type: ignore
from scipy.stats import binomtest
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix

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

# Set directory paths
high_dir = '/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Simulation Results/Rep 1 (100)/High'
medium_dir = '/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Simulation Results/Rep 1 (100)/Medium'
low_dir = '/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Simulation Results/Rep 1 (100)/Low'

# Load haplotypes
high = load_haplotypes(high_dir)
medium = load_haplotypes(medium_dir)
low = load_haplotypes(low_dir)

# User status update
print("Completed setup and loading.")

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

# User status update
print("Completed data pre-processing.")

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

# User status update
print("Completed data augmentation.")

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

# User status update
print("Designed, fitted and saved model.")

"""
Save Tabular Training Metrics
"""
# File logistics
output_directory = '/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Evaluation Logs/Tables/Mirroring'
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
Save Graphical Training Metrics
"""
# Define the directory and file name
output_directory = '/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Evaluation Logs/Images/Mirroring'
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

"""
Model Testing
"""
# Load model
model = load_model('cnn.h5')

# Load test batches
test_data_1 = load_haplotypes('/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Simulation Results/Test Batches/Rep 6/High')
test_data_2 = load_haplotypes('/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Simulation Results/Test Batches/Rep 6/Medium')
test_data_3 = load_haplotypes('/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Simulation Results/Test Batches/Rep 6/Low')

# Define pre-processing function
def process_seqs(batch):
    # Instantize padding length error
    class PaddingError(Exception):
        pass
    # Function logic
    processed_seqs = []
    for sim in batch:
        temp = []
        for ind_seq in sim:
            if len(ind_seq) > 4882:
                x = ind_seq[:4882]
                temp.append(x)
            else:
                padding = 4882 - len(ind_seq)
                if padding < 0:
                    raise PaddingError("padding length is negative")
                else:
                    x = np.pad(array=ind_seq, pad_width=(0,padding), mode='constant', constant_values=-1)
                    temp.append(x)
        processed_seqs.append(temp)
    # Convert to numpy array
    return(np.array(processed_seqs))

# Preprocess test batches
test_data_1 = process_seqs(test_data_1)
test_data_2 = process_seqs(test_data_2)
test_data_3 = process_seqs(test_data_3)

# Build test dataset
test_data = np.concatenate((test_data_1, test_data_2, test_data_3), axis=0)

# Fit test data to tensorflow format
X = tf.constant(test_data)
X = tf.reshape(X, [X.shape[0], X.shape[1], X.shape[2], 1])

# Deploy model for prediction
preds = model.predict(X)

"""
Testing Evaluation Metrics
"""
# Specify directory to save metrics
testing_metrics_file_path = '/Users/jeremiahmushtaq/Documents/University/MSc Research Project/CNN Testing Metrics'

# Define labels
predicted_labels = np.argmax(preds, axis=1) # High=0, Medium=1, Low=2
true_labels = np.array(
    [0]*100 +
    [1]*100 +
    [2]*100
)

# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Visulaise confusion matrix
plt.figure(figsize=(13, 9))
heatmap = sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', # Create the heatmap with larger annotation font size
                      xticklabels=['High', 'Medium', 'Low'],
                      yticklabels=['High', 'Medium', 'Low'],
                      annot_kws={"size": 30})
colorbar = heatmap.collections[0].colorbar # Colour bar settings
colorbar.ax.tick_params(labelsize=24)
plt.xlabel('Predicted Labels', fontsize=30, labelpad=20, weight='bold') # Axes title settings
plt.ylabel('True Labels', fontsize=30, labelpad=20, weight='bold')
plt.xticks(fontsize=24) # Tick label settings
plt.yticks(fontsize=24)
testing_metrics_file_name = 'confusion matrix cnn.png'
full_metrics_path = os.path.join(testing_metrics_file_path, testing_metrics_file_name)
plt.savefig(full_metrics_path)

# Class-wise precision, recall, f1-score
precision = precision_score(true_labels, predicted_labels, average=None)
recall = recall_score(true_labels, predicted_labels, average=None)
f1 = f1_score(true_labels, predicted_labels, average=None)

# Round-off decimal numbers
precision = np.round(precision, 2)
recall = np.round(recall, 2)
f1 = np.round(f1, 2)

# Create class-wise testing metrics dataframe 
df = pd.DataFrame({
    'Migration Rate Class': ['High', 'Medium', 'Low'],
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
})

# Save results
testing_metrics_file_name = 'class-wise testing metrics.tsv'
full_metrics_path = os.path.join(testing_metrics_file_path, testing_metrics_file_name)
df.to_csv(full_metrics_path, sep='\t', index=False)

# Macro-averaged precision, recall, f1-score
macro_precision = precision_score(true_labels, predicted_labels, average='macro')
macro_recall = recall_score(true_labels, predicted_labels, average='macro')
macro_f1 = f1_score(true_labels, predicted_labels, average='macro')

# Create macro testing metrics dataframe
df = pd.DataFrame({
    'Precision': [f"{macro_precision:.2f}"],
    'Recall': [f"{macro_recall:.2f}"],
    'F1-Score': [f"{macro_f1:.2f}"]
})

# Save results
testing_metrics_file_name = 'macro testing metrics.tsv'
full_metrics_path = os.path.join(testing_metrics_file_path, testing_metrics_file_name)
df.to_csv(full_metrics_path, sep='\t', index=False)

"""
Binomial Testing
"""

# Calculate the number of correct predictions
correct_predictions = np.sum(predicted_labels == true_labels)
total_predictions = len(true_labels)

# Baseline accuracy (random guessing for 3 classes)
baseline_accuracy = 1/3

# Perform binomial test
binom_test = binomtest(correct_predictions, total_predictions, baseline_accuracy, alternative='greater')
test_accuracy = correct_predictions/total_predictions*100

# Check if the result is statistically significant
alpha = 0.05
if binom_test.pvalue < alpha:
    outcome = "Significant"
else:
    outcome = "Insignficant"

# Save results
metric_titles = ('Correct Predictions', 'Total Predictions', 'Model Accuracy', 'Baseline Accuracy', 'P-value', 'Significance Threshold', 'Is Result Significant?')
metric_values = [correct_predictions, total_predictions, f"{test_accuracy:.2f}%", f"{baseline_accuracy*100:.2f}%", f"{binom_test.pvalue:.2f}", alpha, outcome]
metric_titles
df = pd.DataFrame({
    'Evaluation Metric': metric_titles,
    'Value': metric_values
})
testing_metrics_file_name = 'binomial testing.tsv'
full_metrics_path = os.path.join(testing_metrics_file_path, testing_metrics_file_name)
df.to_csv(full_metrics_path, sep='\t', index=False)