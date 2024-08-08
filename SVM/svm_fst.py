# Import necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

# Import data
os.chdir('/Users/jeremiahmushtaq/Documents/University/MSc Research Project/Simulation Results')
data = pd.read_csv('fst_sims.txt', sep='\t')

# Remove negative FSTs and prepare data for labelling
data = data[data['FST'] >= 0]
nrows_low = len(data[data['Migration_Rate'] == 1e-09])
nrows_medium = len(data[data['Migration_Rate'] == 0.1])
nrows_high = len(data[data['Migration_Rate'] == 0.9])

# Label data
X = data.drop('Migration_Rate', axis=1)
labels = np.array (['low'] * nrows_low + ['medium'] * nrows_medium + ['high'] * nrows_high)
data['labels'] = labels
y = data['labels']

# Confusion matrix bootstrap validation
n_bootstrap_iterations = 1000
conf_matrices_boot = []

for i in range(n_bootstrap_iterations):
    # Create sample
    X_bootstrap, y_bootstrap = resample(X, y, replace=True)

    # Define train and test split
    X_train, X_test, y_train, y_test = train_test_split(X_bootstrap, y_bootstrap, test_size=0.3, random_state=None)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train classifier
    svm_classifier = SVC(kernel='rbf')
    svm_classifier.fit(X_train_scaled, y_train)

    # Model testing
    y_pred = svm_classifier.predict(X_test_scaled)

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['low', 'medium', 'high'])
    conf_matrices_boot.append(conf_matrix)

# Convert list of confusion matrices to a numpy array for easier manipulation
conf_matrices_boot = np.array(conf_matrices_boot)

# Calculate average confusion matrix
average_conf_matrix = np.mean(conf_matrices_boot, axis=0)

# Visualise and save confusion matrix
plt.figure(figsize=(13, 9))
heatmap = sns.heatmap(average_conf_matrix, annot=True, fmt='.2f', cmap='Blues', # Create the heatmap with larger annotation font size
                      xticklabels=['Low', 'Medium', 'High'],
                      yticklabels=['Low', 'Medium', 'High'],
                      annot_kws={"size": 30})
colorbar = heatmap.collections[0].colorbar # Colour bar settings
colorbar.ax.tick_params(labelsize=24)
plt.xlabel('Predicted Labels', fontsize=30, labelpad=20, weight='bold') # Axes title settings
plt.ylabel('True Labels', fontsize=30, labelpad=20, weight='bold')
plt.xticks(fontsize=24) # Tick label settings
plt.yticks(fontsize=24)
plt.savefig('/Users/jeremiahmushtaq/Documents/University/MSc Research Project/SVM/confusion matrix svm.png')

# Model accuracy bootstrap validation
n_bootstrap_iterations = 1000
boot_accuracies = []

for i in range(n_bootstrap_iterations):
    # Create sample
    X_bootstrap, y_bootstrap = resample(X, y, replace=True)

    # Define train and test split
    X_train, X_test, y_train, y_test = train_test_split(X_bootstrap, y_bootstrap, test_size=0.3, random_state=None)

    # Standarise features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train classifier
    svm_classifier = SVC(kernel='rbf')
    svm_classifier.fit(X_train_scaled, y_train)

    # Model testing
    y_pred = svm_classifier.predict(X_test_scaled)

    # Determine model accuracy
    accuracy = accuracy_score(y_test, y_pred)
    boot_accuracies.append(accuracy)

# Calculate average and standard deviation
average_boot_accuracy = np.mean(boot_accuracies)
std_boot_accuracy = np.std(boot_accuracies)

# Permuation testing
n_permutation_iterations = 1000
perm_accuracies = []

for i in range(n_permutation_iterations):
    # Shuffle labels
    y_permuted = np.random.permutation(y)

    # Create sample
    X_bootstrap, y_bootstrap = resample(X, y_permuted, replace=True)

    # Define train and test split
    X_train, X_test, y_train, y_test = train_test_split(X_bootstrap, y_bootstrap, test_size=0.3, random_state=None)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train classifier
    svm_classifier = SVC(kernel='rbf')
    svm_classifier.fit(X_train_scaled, y_train)

    # Model testing
    y_pred = svm_classifier.predict(X_test_scaled)

    # Determine model performance
    accuracy = accuracy_score(y_test, y_pred)
    perm_accuracies.append(accuracy)

# Calculate average and standard deviation
average_perm_accuracy = np.mean(perm_accuracies)
std_perm_accuracy = np.std(perm_accuracies)

# Compare observed accuracy with permutation distribution
p_value = np.mean(np.array(perm_accuracies) >= average_boot_accuracy)

# Determine significance
if p_value < 0.05:
    outcome = 'Significant'
else:
    outcome = 'Insignificant'

# Output metrics table
metric_titles = ['Average Bootstrap Accuracy', 'Bootstrap Standard Deviation', 'Average Permutation Accuracy', 'Permutation Standard Deviation', 'P-value', 'Is Result Significant?']
metric_values = [f"{average_boot_accuracy*100:.2f}%", f"{std_boot_accuracy*100:.2f}%", f"{average_perm_accuracy*100:.2f}%", f"{std_perm_accuracy*100:.2f}%", f"{p_value:.2f}", outcome]
metrics_df = pd.DataFrame({
    'Evaluation Metric': metric_titles,
    'Value': metric_values
})
metrics_path = '/Users/jeremiahmushtaq/Documents/University/MSc Research Project/SVM'
metrics_file_name = 'metrics svm.tsv'
full_metrics_path = os.path.join(metrics_path, metrics_file_name)
metrics_df.to_csv(full_metrics_path, index=False)