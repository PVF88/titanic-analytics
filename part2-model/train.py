"""
PART 2 — Predictive Model
Trains a survival prediction model on Titanic data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import time
import os

os.makedirs('model', exist_ok=True)

def load_data():
    """Load cleaned data"""
    df = pd.read_csv('../part1-data-cleaning/data_clean/titanic_clean.csv')
    return df

def train_model(X_train, y_train):
    """Train model - MULTIPLE PERFORMANCE ISSUES"""
    
    # ISSUE 1: No hyperparameter optimization, just using defaults
    # ISSUE 2: Training on full dataset without early stopping validation
    print("Training model...")
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    return clf

def evaluate_model(clf, X_train, y_train, X_test, y_test):
    """Evaluate model - INEFFICIENT SCORING"""
    
    # ISSUE 3: Scoring on train and test separately instead of batch prediction
    print("Evaluating...")
    
    # Recalculate predictions even though they were computed during training
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # ISSUE 4: Getting predictions multiple times
    train_pred = clf.predict(X_train)
    train_pred_proba = clf.predict_proba(X_train)  # Redundant calculation
    
    test_pred = clf.predict(X_test)
    test_pred_proba = clf.predict_proba(X_test)  # Redundant calculation
    
    return {
        'train_score': train_score,
        'test_score': test_score,
        'train_pred': train_pred,
        'test_pred': test_pred
    }

def main():
    start_time = time.time()
    
    # Load data
    df = load_data()
    print(f"Loaded data shape: {df.shape}")
    
    # ISSUE 5: Not selecting features, using all columns
    X = df.drop('survived', axis=1) if 'survived' in df.columns else df
    y = df['survived'] if 'survived' in df.columns else None
    
    if y is None:
        print("ERROR: No target column found")
        return
    
    # ISSUE 6: Simple train/test split, no cross-validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Train
    clf = train_model(X_train, y_train)
    
    # ISSUE 7: Pickle is slow and unsafe - not using joblib
    print("Saving model...")
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    
    # Evaluate
    results = evaluate_model(clf, X_train, y_train, X_test, y_test)
    
    elapsed = time.time() - start_time
    print(f"\nTotal training time: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
