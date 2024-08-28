"""
This file implements the testing/prediction phase of the CNN. 
"""

# Import dependancies
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras.models import load_model  # type: ignore
from scipy.stats import binomtest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from functions import load_haplotypes, process_seqs

"""
Model Testing
"""
# Specifiy path to model
os.chdir('/Users/jeremiahmushtaq/Documents/University/MSc Research Project/CNN')

# Load model
model = load_model('cnn.h5')

# Load test batches
test_data_1 = load_haplotypes('/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Simulation Results/CNN Testing Data/Rep 6/High')
test_data_2 = load_haplotypes('/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Simulation Results/CNN Testing Data/Rep 6/Medium')
test_data_3 = load_haplotypes('/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Simulation Results/CNN Testing Data/Rep 6/Low')

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
Evaluation Metrics
"""
# Specify directory to save metrics
testing_metrics_file_path = '/Users/jeremiahmushtaq/Documents/University/MSc Research Project/CNN'

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
testing_metrics_file_name = 'confusion_matrix_cnn.png'
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
df_1 = pd.DataFrame({
    'Migration Rate Class': ['High', 'Medium', 'Low'],
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
})

# Macro-averaged precision, recall, f1-score
macro_precision = precision_score(true_labels, predicted_labels, average='macro')
macro_recall = recall_score(true_labels, predicted_labels, average='macro')
macro_f1 = f1_score(true_labels, predicted_labels, average='macro')

# Create macro testing metrics dataframe
df_2 = pd.DataFrame({
    'Migration Rate Class': ['Macro Average'],
    'Precision': [f"{macro_precision:.2f}"],
    'Recall': [f"{macro_recall:.2f}"],
    'F1-Score': [f"{macro_f1:.2f}"]
})

# Merge dataframes
df = pd.concat([df_1, df_2], ignore_index=True)

# Save results
testing_metrics_file_name = 'testing_metrics_cnn.tsv'
full_metrics_path = os.path.join(testing_metrics_file_path, testing_metrics_file_name)
df.to_csv(full_metrics_path, sep='\t', index=True)

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
testing_metrics_file_name = 'binomial_testing.tsv'
full_metrics_path = os.path.join(testing_metrics_file_path, testing_metrics_file_name)
df.to_csv(full_metrics_path, sep='\t', index=False)