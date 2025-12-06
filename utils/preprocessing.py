"""
Data Preprocessing Utilities for Mobile Market Segmenter

This module contains helper functions for:
- Feature scaling
- Data encoding
- Missing value imputation
- Feature engineering

Author: Mayank Goyal
Project: Mobile Market Segmenter
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_and_clean_data(filepath, target_column='User Behavior Class'):
    """
    Load and clean the mobile user dataset.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    target_column : str
        Name of the target variable column
    
    Returns:
    --------
    X : DataFrame
        Features
    y : Series
        Target variable
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Handle missing values
    df = df.dropna()
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y


def encode_categorical_features(X, categorical_columns=None):
    """
    One-hot encode categorical features.
    
    Parameters:
    -----------
    X : DataFrame
        Feature dataset
    categorical_columns : list, optional
        List of categorical column names
    
    Returns:
    --------
    X_encoded : DataFrame
        Encoded features
    """
    if categorical_columns is None:
        # Auto-detect categorical columns
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    
    # One-hot encoding
    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=False)
    
    return X_encoded


def scale_features(X_train, X_test):
    """
    Standardize numerical features using StandardScaler.
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    X_test : DataFrame
        Testing features
    
    Returns:
    --------
    X_train_scaled : ndarray
        Scaled training features
    X_test_scaled : ndarray
        Scaled testing features
    scaler : StandardScaler
        Fitted scaler object
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def encode_target(y_train, y_test):
    """
    Label encode the target variable.
    
    Parameters:
    -----------
    y_train : Series
        Training target
    y_test : Series
        Testing target
    
    Returns:
    --------
    y_train_encoded : ndarray
        Encoded training target
    y_test_encoded : ndarray
        Encoded testing target
    le : LabelEncoder
        Fitted label encoder
    """
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    return y_train_encoded, y_test_encoded, le


def prepare_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : DataFrame
        Features
    y : Series
        Target variable
    test_size : float
        Proportion of test set (default: 0.2)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : DataFrames/Series
        Split datasets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Ensure balanced classes
    )
    
    return X_train, X_test, y_train, y_test


def full_preprocessing_pipeline(filepath, target_column='User Behavior Class', test_size=0.2):
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    filepath : str
        Path to dataset
    target_column : str
        Target variable name
    test_size : float
        Test set size
    
    Returns:
    --------
    X_train_scaled : ndarray
        Preprocessed training features
    X_test_scaled : ndarray
        Preprocessed testing features
    y_train_encoded : ndarray
        Encoded training target
    y_test_encoded : ndarray
        Encoded testing target
    scaler : StandardScaler
        Fitted scaler
    le : LabelEncoder
        Fitted label encoder
    feature_names : list
        List of feature column names
    """
    # Step 1: Load data
    X, y = load_and_clean_data(filepath, target_column)
    
    # Step 2: Split data
    X_train, X_test, y_train, y_test = prepare_train_test_split(X, y, test_size)
    
    # Step 3: Encode categorical features
    X_train_encoded = encode_categorical_features(X_train)
    X_test_encoded = encode_categorical_features(X_test)
    
    # Align columns (ensure test has same columns as train)
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    
    # Step 4: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_encoded, X_test_encoded)
    
    # Step 5: Encode target
    y_train_encoded, y_test_encoded, le = encode_target(y_train, y_test)
    
    # Feature names
    feature_names = X_train_encoded.columns.tolist()
    
    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, scaler, le, feature_names


if __name__ == "__main__":
    # Example usage
    print("Preprocessing utilities loaded successfully!")
    print("Available functions:")
    print("  - load_and_clean_data()")
    print("  - encode_categorical_features()")
    print("  - scale_features()")
    print("  - encode_target()")
    print("  - prepare_train_test_split()")
    print("  - full_preprocessing_pipeline()")
