import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_graphviz
import graphviz
import os

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Load and preprocess the dataset
def load_and_preprocess_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(url, names=columns)
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    df['ca'] = pd.to_numeric(df['ca'])
    df['thal'] = pd.to_numeric(df['thal'])
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    return df

# Step 2: Train and visualize Decision Tree
def train_decision_tree(X_train, X_test, y_train, y_test, max_depth=None):
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)
    
    # Calculate accuracy
    train_pred = dt.predict(X_train)
    test_pred = dt.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    # Visualize the tree with error handling
    try:
        dot_data = export_graphviz(dt, out_file=None, 
                                 feature_names=X_train.columns,
                                 class_names=['No Disease', 'Disease'],
                                 filled=True, rounded=True)
        graph = graphviz.Source(dot_data)
        graph.render(f"decision_tree_depth_{max_depth or 'none'}", format='png', cleanup=True)
        print(f"Decision tree visualization saved as decision_tree_depth_{max_depth or 'none'}.png")
    except Exception as e:
        print(f"Error generating decision tree visualization: {e}")
        print("Ensure Graphviz is installed and added to your system PATH.")
        print("Continuing without visualization...")
    
    return dt, train_acc, test_acc

# Step 3: Train Random Forest
def train_random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    train_pred = rf.predict(X_train)
    test_pred = rf.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    return rf, train_acc, test_acc

# Step 4: Analyze feature importance
def plot_feature_importance(model, features, model_name):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importance - {model_name}")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_feature_importance.png")
    plt.close()

# Step 5: Perform cross-validation
def perform_cross_validation(model, X, y, model_name):
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{model_name} Cross-validation scores: {scores}")
    print(f"{model_name} Average CV score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

def main():
    df = load_and_preprocess_data()
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    
    print("Decision Tree Analysis:")
    for depth in [3, 5, None]:
        dt, train_acc, test_acc = train_decision_tree(X_train, X_test, y_train, y_test, max_depth=depth)
        print(f"Max Depth {depth}:")
        print(f"Training Accuracy: {train_acc:.3f}")
        print(f"Testing Accuracy: {test_acc:.3f}")
        print("Classification Report:")
        print(classification_report(y_test, dt.predict(X_test)))
        perform_cross_validation(dt, X, y, f"Decision Tree (depth={depth})")
        plot_feature_importance(dt, X.columns, f"Decision Tree (depth={depth})")
        print("\n")
    
    print("Random Forest Analysis:")
    rf, train_acc, test_acc = train_random_forest(X_train, X_test, y_train, y_test)
    print(f"Training Accuracy: {train_acc:.3f}")
    print(f"Testing Accuracy: {test_acc:.3f}")
    print("Classification Report:")
    print(classification_report(y_test, rf.predict(X_test)))
    perform_cross_validation(rf, X, y, "Random Forest")
    plot_feature_importance(rf, X.columns, "Random Forest")

if __name__ == "__main__":
    main()