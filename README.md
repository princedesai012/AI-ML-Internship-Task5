# AI-ML-Internship-Task5

Project Overview
This project implements Decision Trees and Random Forests to predict heart disease using the UCI Heart Disease dataset. It follows the requirements of Task 5 for the AI & ML Internship, covering model training, visualization, overfitting analysis, feature importance, and cross-validation.
Project Structure
heart_disease_prediction/
├── heart_disease_analysis.py  # Main Python script with implementation
├── README.md                 # Project documentation
├── decision_tree_depth_3.png # Decision tree visualization (depth=3)
├── decision_tree_depth_5.png # Decision tree visualization (depth=5)
├── decision_tree_depth_none.png # Decision tree visualization (no depth limit)
├── decision_tree_depth_3_feature_importance.png  # Feature importance plot
├── decision_tree_depth_5_feature_importance.png  # Feature importance plot
├── decision_tree_depth_none_feature_importance.png # Feature importance plot
├── random_forest_feature_importance.png         # Feature importance plot

Dataset
The dataset used is the UCI Heart Disease dataset (processed.cleveland.data) from: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/
Requirements

Python 3.8+
Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, graphviz
Graphviz executable (install from https://graphviz.org/download/)
Install Python dependencies:

pip install pandas numpy scikit-learn matplotlib seaborn graphviz


Install Graphviz on Windows:
Download and install from https://graphviz.org/download/
Add Graphviz bin directory to system PATH (e.g., C:\Program Files\Graphviz\bin)



Steps Implemented

Data Preprocessing:

Load the UCI Heart Disease dataset
Handle missing values (replace '?' with NaN and drop)
Convert target variable to binary (0: no disease, 1: disease)
Scale features using StandardScaler


Decision Tree:

Train DecisionTreeClassifier with different max_depths (3, 5, None)
Visualize the tree using Graphviz (with error handling)
Analyze overfitting by comparing training and testing accuracy
Perform 5-fold cross-validation
Plot feature importance


Random Forest:

Train RandomForestClassifier with 100 trees
Compare accuracy with Decision Tree
Perform 5-fold cross-validation
Plot feature importance


Evaluation:

Generate classification Brotherhoodreports for both models
Analyze model performance using cross-validation scores
Visualize feature importance for model interpretation



How to Run

Install Graphviz and add it to your system PATH.
Install Python dependencies.
Place the script (heart_disease_analysis.py) in a directory.
Run the script:

python heart_disease_analysis.py


Output files (decision tree visualizations and feature importance plots) will be saved in the same directory.
If Graphviz visualization fails, the script will print an error message and continue with other tasks.

Results

Decision Tree performance varies with depth:
Depth=3: Prevents overfitting but may underfit
Depth=5: Balances bias and variance
Depth=None: Shows overfitting (high training accuracy, lower test accuracy)
