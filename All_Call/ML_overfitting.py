import numpy as np
import pandas as pd
import scipy
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import copy
import logging
import missingno
from collections import Counter

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    learning_curve,
    KFold
)
from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelBinarizer, 
    MultiLabelBinarizer,
    MaxAbsScaler,
    QuantileTransformer,
    PowerTransformer,
    Normalizer
)
from category_encoders import TargetEncoder
from sklearn.feature_extraction import (
    DictVectorizer, 
    FeatureHasher
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from imblearn.over_sampling import (
    SMOTE,
    RandomOverSampler
)
from imblearn.under_sampling import (
    TomekLinks, 
    RandomUnderSampler
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    NearestNeighbors
)
from sklearn.svm import (
    SVC,
    SVR
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    RandomForestRegressor,
    BaggingRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    StackingRegressor
)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from scipy.stats import uniform, randint
from fast_ml.model_development import train_valid_test_split
from typing import (
    Any,
    Literal,
    Union,
    List,
    Optional,
    Dict,
    Tuple
)
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut, LeavePOut, RepeatedKFold, TimeSeriesSplit, cross_val_score, learning_curve
)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris


import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def get_cross_validator(cv_type: Literal['KFold', 'StratifiedKFold', 'LeaveOneOut', 'LeavePOut', 'RepeatedKFold', 'TimeSeriesSplit'] = 'KFold', 
                        cv=5, shuffle=True, random_state=42, LeavePOut_p=2, RepeatedKFold_n_repeats=10):
    if cv_type == 'KFold':
        type_cross_valid = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
    elif cv_type == 'StratifiedKFold':
        type_cross_valid = StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
    elif cv_type == 'LeaveOneOut':
        type_cross_valid = LeaveOneOut()
    elif cv_type == 'LeavePOut':
        type_cross_valid = LeavePOut(p=LeavePOut_p)
    elif cv_type == 'RepeatedKFold':
        type_cross_valid = RepeatedKFold(n_splits=cv, n_repeats=RepeatedKFold_n_repeats, random_state=random_state)
    elif cv_type == 'TimeSeriesSplit':
        type_cross_valid = TimeSeriesSplit(n_splits=cv)
    else:
        raise ValueError("Invalid cv_type. Choose from 'KFold', 'StratifiedKFold', 'LeaveOneOut', 'LeavePOut', 'RepeatedKFold', 'TimeSeriesSplit'.")
    return type_cross_valid


def Check_Overfitting_Classification(
    model,
    x: np.ndarray,
    y: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    learning_curve_scoring: Literal['accuracy', 'precision', 'recall', 'f1', 'roc_auc'] = 'accuracy',
    cv_type: Literal['KFold', 'StratifiedKFold', 'LeaveOneOut', 'LeavePOut', 'RepeatedKFold', 'TimeSeriesSplit'] = 'KFold',
    cv: int = 5,
    cv_scoring: Literal['accuracy', 'precision', 'recall', 'f1', 'roc_auc'] = 'accuracy',
    shuffle: bool = True,
    LeavePOut_p: int = 2,
    RepeatedKFold_n_repeats: int = 10,
    random_state: int = 42,
    plot: bool = True
) -> Dict[str, Any]:
    """
    Evaluate the performance of a model to check for overfitting.

    Parameters:
    - model: The machine learning model to evaluate.
    - x: Feature set for cross-validation.
    - y: Target set for cross-validation.
    - x_train: Training feature set.
    - y_train: Training target set.
    - x_valid: Validation feature set.
    - y_valid: Validation target set.
    - learning_curve_scoring: Scoring metric for learning curve (default is 'accuracy').
    - cv_type: Type of cross-validation ('KFold', 'StratifiedKFold', 'LeaveOneOut', 'LeavePOut', 'RepeatedKFold', 'TimeSeriesSplit').
    - cv: Number of cross-validation folds (default is 5).
    - cv_scoring: Scoring metric for cross-validation (default is 'accuracy').
    - shuffle: Whether to shuffle the data before splitting (default is True).
    - LeavePOut_p: Number of samples to leave out in LeavePOut (default is 2).
    - RepeatedKFold_n_repeats: Number of repeats in RepeatedKFold (default is 10).
    - random_state: Random seed for reproducibility (default is 42).
    - plot: Whether to plot the learning and ROC curves (default is True).

    Returns:
    - A dictionary containing various evaluation metrics.
    """
    
    y_train_pred = model.predict(x_train)
    y_valid_pred = model.predict(x_valid)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    valid_precision = precision_score(y_valid, y_valid_pred, average='weighted')
    
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    valid_recall = recall_score(y_valid, y_valid_pred, average='weighted')
    
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    valid_f1 = f1_score(y_valid, y_valid_pred, average='weighted')
    
    train_mcc = matthews_corrcoef(y_train, y_train_pred)
    valid_mcc = matthews_corrcoef(y_valid, y_valid_pred)
    
    train_balanced_acc = balanced_accuracy_score(y_train, y_train_pred)
    valid_balanced_acc = balanced_accuracy_score(y_valid, y_valid_pred)
    
    conf_matrix = confusion_matrix(y_valid, y_valid_pred)
    
    if hasattr(model, "predict_proba"):
        if len(np.unique(y)) == 2:
            y_val_prob = model.predict_proba(x_valid)[:, 1]
            roc_auc = roc_auc_score(y_valid, y_val_prob)
            fpr, tpr, _ = roc_curve(y_valid, y_val_prob)
        else:
            y_val_prob = model.predict_proba(x_valid)
            roc_auc = roc_auc_score(y_valid, y_val_prob, multi_class='ovr')
            fpr, tpr = None, None
    else:
        roc_auc = None
        fpr, tpr = None, None
    
    if cv_type == 'KFold':
        type_cross_valid = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
    elif cv_type == 'StratifiedKFold':
        type_cross_valid = StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
    elif cv_type == 'LeaveOneOut':
        type_cross_valid = LeaveOneOut()
    elif cv_type == 'LeavePOut':
        type_cross_valid = LeavePOut(p=LeavePOut_p)
    elif cv_type == 'RepeatedKFold':
        type_cross_valid = RepeatedKFold(n_splits=cv, n_repeats=RepeatedKFold_n_repeats, random_state=random_state)
    elif cv_type == 'TimeSeriesSplit':
        type_cross_valid = TimeSeriesSplit(n_splits=cv)
    else:
        raise ValueError("Invalid cv_type. Choose from 'KFold', 'StratifiedKFold', 'LeaveOneOut', 'LeavePOut', 'RepeatedKFold', 'TimeSeriesSplit'.")
    
    cv_scores = cross_val_score(model, x, y, cv=type_cross_valid, scoring=cv_scoring)
    
    # Compute the learning curves
    train_sizes, train_scores, valid_scores = learning_curve(model, x, y, cv=type_cross_valid, scoring=learning_curve_scoring, n_jobs=-1, random_state=random_state)
    train_scores_mean = np.mean(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)
    
    # # Compute the learning curves
    # train_sizes_t, train_scores_t, valid_scores_t = learning_curve(model, x_train, y_train, cv=type_cross_valid, scoring=learning_curve_scoring, n_jobs=-1, random_state=random_state)
    # # Calculate the mean and standard deviation for training and validation scores
    # train_mean_t = np.mean(train_scores_t, axis=1)
    # train_std_t = np.std(train_scores_t, axis=1)
    # val_mean_t = np.mean(valid_scores_t, axis=1)
    # val_std_t = np.std(valid_scores_t, axis=1)
    
    print('Accuracy:')
    print(f'Training Accuracy: {train_accuracy:.4f}')
    print(f'Validation Accuracy: {valid_accuracy:.4f}')
    
    print('\nPrecision:')
    print(f'Training Precision: {train_precision:.4f}')
    print(f'Validation Precision: {valid_precision:.4f}')
    
    print('\nRecall:')
    print(f'Training Recall: {train_recall:.4f}')
    print(f'Validation Recall: {valid_recall:.4f}')
    
    print('\nF1-Score:')
    print(f'Training F1-Score: {train_f1:.4f}')
    print(f'Validation F1-Score: {valid_f1:.4f}')
    
    print('\nMCC:')
    print(f'Training MCC: {train_mcc:.4f}')
    print(f'Validation MCC: {valid_mcc:.4f}')
    
    print('\nBalanced Accuracy:')
    print(f'Training Balanced Accuracy: {train_balanced_acc:.4f}')
    print(f'Validation Balanced Accuracy: {valid_balanced_acc:.4f}')
    
    print('\nConfusion Matrix:')
    print(f'Validation Confusion Matrix:\n{conf_matrix}')
    
    print('\nCross-Validation(CV):')
    print(f'Cross-Validation Scores: {cv_scores}')
    print(f'Cross-Validation Mean Score: {cv_scores.mean():.4f}')
    
    if plot:
        # Plot the learning curves
        plt.figure()
        plt.plot(train_sizes, train_scores_mean, "r-+", label='Training Accuracy')
        plt.plot(train_sizes, valid_scores_mean, "b-*", label='Validation Accuracy')
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color='r', alpha=0.25)
        plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, color='b', alpha=0.25)
        plt.xlabel('Training Size')
        plt.ylabel('Accuracy')
        plt.title('Learning Curve')
        plt.legend()
        plt.show()
        
        
        # # Plot the learning curves
        # plt.figure(figsize=(10, 6))
        # plt.plot(train_sizes_t, train_mean_t, 'o-', color='r', label='Training score')
        # plt.plot(train_sizes_t, val_mean_t, 'o-', color='g', label='Validation score')
        # plt.fill_between(train_sizes_t, train_mean_t - train_std_t, train_mean_t + train_std_t, color='r', alpha=0.1)
        # plt.fill_between(train_sizes_t, val_mean_t - val_std_t, val_mean_t + val_std_t, color='g', alpha=0.1)
        # plt.title('Learning Curves')
        # plt.xlabel('Training Set Size')
        # plt.ylabel('Accuracy')
        # plt.legend(loc='best')
        # plt.grid()
        # plt.show()
        
        
        if roc_auc is not None and fpr is not None and tpr is not None:
            print(f'ROC AUC: {roc_auc:.4f}')
            plt.figure()
            plt.plot(fpr, tpr, "g-o", label=f'ROC Curve (AUC = {roc_auc:.4f})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.show()
        else:
            print('ROC AUC: Not available for this model')
    
    results = {
        'train_accuracy': train_accuracy,
        'valid_accuracy': valid_accuracy,
        'train_precision': train_precision,
        'valid_precision': valid_precision,
        'train_recall': train_recall,
        'valid_recall': valid_recall,
        'train_f1': train_f1,
        'valid_f1': valid_f1,
        'train_mcc': train_mcc,
        'valid_mcc': valid_mcc,
        'train_balanced_acc': train_balanced_acc,
        'valid_balanced_acc': valid_balanced_acc,
        'conf_matrix': conf_matrix,
        'roc_auc': roc_auc,
        'cv_scores': cv_scores,
        'train_sizes': train_sizes,
        'train_scores_mean': train_scores_mean,
        'valid_scores_mean': valid_scores_mean
    }
    # return results


def Check_Overfitting_Regression(
    model,
    x: np.ndarray,
    y: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    learning_curve_scoring: Literal['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'] = 'neg_mean_squared_error',
    cv_type: Literal['KFold', 'StratifiedKFold', 'LeaveOneOut', 'LeavePOut', 'RepeatedKFold', 'TimeSeriesSplit'] = 'KFold',
    cv: int = 5,
    cv_scoring: Literal['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'] = 'neg_mean_squared_error',
    shuffle: bool = True,
    LeavePOut_p: int = 2,
    RepeatedKFold_n_repeats: int = 10,
    random_state: int = 42,
    plot: bool = True
) -> Dict[str, Any]:
    y_train_pred = model.predict(x_train)
    y_valid_pred = model.predict(x_valid)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    valid_mae = mean_absolute_error(y_valid, y_valid_pred)

    train_mse = mean_squared_error(y_train, y_train_pred)
    valid_mse = mean_squared_error(y_valid, y_valid_pred)

    train_rmse = np.sqrt(train_mse)
    valid_rmse = np.sqrt(valid_mse)

    train_r2 = r2_score(y_train, y_train_pred)
    valid_r2 = r2_score(y_valid, y_valid_pred)

    if cv_type == 'KFold':
        type_cross_valid = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
    elif cv_type == 'StratifiedKFold':
        type_cross_valid = StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
    elif cv_type == 'LeaveOneOut':
        type_cross_valid = LeaveOneOut()
    elif cv_type == 'LeavePOut':
        type_cross_valid = LeavePOut(p=LeavePOut_p)
    elif cv_type == 'RepeatedKFold':
        type_cross_valid = RepeatedKFold(n_splits=cv, n_repeats=RepeatedKFold_n_repeats, random_state=random_state)
    elif cv_type == 'TimeSeriesSplit':
        type_cross_valid = TimeSeriesSplit(n_splits=cv)
    else:
        raise ValueError("Invalid cv_type. Choose from 'KFold', 'StratifiedKFold', 'LeaveOneOut', 'LeavePOut', 'RepeatedKFold', 'TimeSeriesSplit'.")

    cv_scores = cross_val_score(model, x, y, cv=type_cross_valid, scoring=cv_scoring)

    train_sizes, train_scores, valid_scores = learning_curve(model, x, y, cv=type_cross_valid, scoring=learning_curve_scoring, n_jobs=-1, random_state=random_state)
    
    if learning_curve_scoring in ['neg_mean_squared_error', 'neg_mean_absolute_error']:
        train_scores_mean = -np.mean(train_scores, axis=1)
        valid_scores_mean = -np.mean(valid_scores, axis=1)
    else:
        train_scores_mean = np.mean(train_scores, axis=1)
        valid_scores_mean = np.mean(valid_scores, axis=1)
    
    
    # # Compute the learning curves
    # train_sizes_t, train_scores_t, valid_scores_t = learning_curve(model, x_train, y_train, cv=type_cross_valid, scoring=learning_curve_scoring, n_jobs=-1, random_state=random_state)
    # # Calculate the mean and standard deviation for training and validation scores
    # if learning_curve_scoring in ['neg_mean_squared_error', 'neg_mean_absolute_error']:
    #     # Convert scores to positive
    #     train_scores_t = -train_scores_t
    #     valid_scores_t = -valid_scores_t
    # train_mean_t = np.mean(train_scores_t, axis=1)
    # train_std_t = np.std(train_scores_t, axis=1)
    # val_mean_t = np.mean(valid_scores_t, axis=1)
    # val_std_t = np.std(valid_scores_t, axis=1)
    

    print('Mean Absolute Error (MAE):')
    print(f'Training MAE: {train_mae:.4f}')
    print(f'Validation MAE: {valid_mae:.4f}')

    print('\nMean Squared Error (MSE):')
    print(f'Training MSE: {train_mse:.4f}')
    print(f'Validation MSE: {valid_mse:.4f}')

    print('\nRoot Mean Squared Error (RMSE):')
    print(f'Training RMSE: {train_rmse:.4f}')
    print(f'Validation RMSE: {valid_rmse:.4f}')

    print('\nR² Score:')
    print(f'Training R²: {train_r2:.4f}')
    print(f'Validation R²: {valid_r2:.4f}')

    print('\nCross-Validation (CV):')
    if cv_scoring in ['neg_mean_squared_error', 'neg_mean_absolute_error']:
        cv_scores = -cv_scores
        print(f'Cross-Validation Scores: {cv_scores}')
        print(f'Cross-Validation Mean Score: {cv_scores.mean():.4f}')
    else:
        print(f'Cross-Validation Scores: {cv_scores}')
        print(f'Cross-Validation Mean Score: {cv_scores.mean():.4f}')

    if plot:
        plt.figure()
        plt.plot(train_sizes, train_scores_mean, label='Training Score')
        plt.plot(train_sizes, valid_scores_mean, label='Validation Score')
        plt.xlabel('Training Size')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend()
        plt.show()
        
        # # Plot the learning curves
        # plt.figure(figsize=(10, 6))
        # plt.plot(train_sizes_t, train_mean_t, 'o-', color='r', label='Training score')
        # plt.plot(train_sizes_t, val_mean_t, 'o-', color='g', label='Validation score')
        # plt.fill_between(train_sizes_t, train_mean_t - train_std_t, train_mean_t + train_std_t, color='r', alpha=0.1)
        # plt.fill_between(train_sizes_t, val_mean_t - val_std_t, val_mean_t + val_std_t, color='g', alpha=0.1)
        # plt.title('Learning Curves')
        # plt.xlabel('Training Set Size')
        # plt.ylabel('Accuracy')
        # plt.legend(loc='best')
        # plt.grid()
        # plt.show()

    results = {
        'train_mae': train_mae,
        'valid_mae': valid_mae,
        'train_mse': train_mse,
        'valid_mse': valid_mse,
        'train_rmse': train_rmse,
        'valid_rmse': valid_rmse,
        'train_r2': train_r2,
        'valid_r2': valid_r2,
        'cv_scores': cv_scores,
        'train_sizes': train_sizes,
        'train_scores_mean': train_scores_mean,
        'valid_scores_mean': valid_scores_mean
    }

    # return results

















































