# **MosiTrak AI**

Welcome to **MosiTrak AI**! This program is designed to classify migration levels in populations of the *Anopheles* mosquito in Africa using advanced Convolutional Neural Networks (CNN) technology.

## **How to Use MosiTrak AI**

### **Using the CNN Model**

#### **Step 1: Download the Repository**
Clone or download the repository to your local machine.

#### **Step 2: Enable Functions**
Run the `functions.py` file to enable the functions required for downstream implementation.

#### **Step 3a: Train the CNN Model**
Run the `cnn_model_training.py` script located in the `CNN` folder to train the model.

- **Note:** Redefine file paths in the script according to your file path preferences.
- **Output:** This should generate:
  - `plot.png`: A graphical representation of the training and validation process.
  - `metrics.tsv`: A tabular representation of the training and validation metrics.

#### **Step 3b: Use a Pre-trained Model**
Alternatively, download a pre-trained version of the model from the following link:

[Pre-trained Model Download](https://drive.google.com/file/d/1jXg_3rkgLA0sAVWjun2BMYd1Dg4tduUN/view?usp=sharing)

#### **Step 4: Test/Deploy the CNN Model**
Run the `cnn_model_testing.py` script located in the `CNN` folder to test or deploy the model for prediction.

- **Note:** Redefine file paths in the script according to your file path preferences.
- **Output:** This should generate:
  - `confusion_matrix_cnn.png`: A graphical representation of the confusion matrix.
  - `testing_metrics_cnn.tsv`: A tabular file with testing metrics.
  - `binomial_testing.tsv`: A file providing information on the statistical significance of the model results.

### **Using the SVM Model**

If you prefer to use the Support Vector Machine (SVM) model instead of the CNN model, follow these steps:

#### **Step 1: Download the Repository**
Clone or download the repository to your local machine.

#### **Step 2: Run the SVM Model**
Run the `svm_fst.py` script in the main folder.

- **Note:** Redefine file paths in the script according to your file path preferences.
- **Output:** This should generate:
  - `confusion_matrix_svm.png`: A graphical representation of the confusion matrix for the SVM model.
  - `metrics_svm.tsv`: A file detailing the outcome and reliability of the model results.

## **Other Tools**

If you wish to simulate your own data for analysis:

- Run the `simulation.py` script and modify the parameters according to your specifications.

## **Miscellaneous Files**

- **Training Evaluation Logs:** The `Training Evaluation Logs` folder contains records of the optimization and hyperparameter tuning process.
- **FST Plot Comparison:** The `fst_plot.R` file can be used to overlap and compare three different migration rates.

## **FYI**
This model was developed for my Masters Research Project. Admitably, the predictive power of the model is low due to some limitations. These limitations and further information about this project found in the `Dissertation.pdf` file in the main folder.