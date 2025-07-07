# Fraud-Detection-System-using-Machine-Learning

## Setup Instructions

1. **Clone or Download the Repository**
   - Clone the GitHub repository or unzip the downloaded `.zip` file into your local environment.

2. **Install Required Libraries**
   Use pip to install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn tensorflow
   ```

3. **Download Dataset**
   - Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Download `creditcard.csv` and place it in the same directory as the notebook

4. **Run the Notebook**
   Open the notebook in Jupyter or VSCode and run each cell sequentially.

---

## Explanation of Approach

1. **Exploratory Data Analysis (EDA)**
   - Loaded the dataset and examined the class distribution (high imbalance).
   - Visualized class counts and checked for missing values.

2. **Feature Engineering**
   - Scaled `Amount` and `Time` features.
   - Dropped and reordered features for better performance.

3. **Imbalance Handling**
   - Applied **SMOTE** to oversample the minority class (fraud cases).

4. **Model Training**
   - Trained a baseline **Random Forest** model.
   - Performed **hyperparameter tuning** with `RandomizedSearchCV`.

5. **Evaluation Metrics**
   - Focused on **Precision**, **Recall**, **F1-score**, and **AUC-ROC** rather than accuracy.
   - Visualized Confusion Matrix, ROC Curve, and Feature Importances.

6. **Bonus Models**
   - **Isolation Forest**: Unsupervised anomaly detection method
   - **Autoencoder**: Neural network trained on normal data to flag anomalies by reconstruction error

---

## Challenges Faced & Solutions

### 1. **Extreme Class Imbalance**
- **Problem**: Only ~0.17% of the dataset consists of fraudulent transactions.
- **Solution**: Applied **SMOTE** for synthetic oversampling and also explored **anomaly detection** methods.

### 2. **Long Training Time for Randomized Search**
- **Problem**: Hyperparameter tuning on Random Forest took very long (over 1000 seconds).
- **Solution**: Limited the number of iterations in `RandomizedSearchCV` and switched to GPU-compatible models like XGBoost (optional improvement).

### 3. **GPU Not Utilized in Scikit-learn**
- **Problem**: Random Forest and other sklearn models do not leverage GPU acceleration.
- **Solution**: Recommended switching to **XGBoost** with `tree_method='gpu_hist'` for faster tuning.

### 4. **False Positives in Anomaly Detection**
- **Problem**: Isolation Forest and Autoencoders often flagged normal transactions as fraud.
- **Solution**: Used reconstruction error thresholding and tuned sensitivity for better precision.
