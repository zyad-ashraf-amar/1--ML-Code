import numpy as np
import pandas as pd
import copy
from scipy import stats
import logging
from typing import (
    Any, 
    Union, 
    List, 
    Optional, 
    Dict, 
    Tuple, 
    Literal
)
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    RobustScaler, 
    MaxAbsScaler, 
    QuantileTransformer, 
    PowerTransformer, 
    Normalizer,
    LabelEncoder,
    OrdinalEncoder,
    LabelBinarizer,
    MultiLabelBinarizer
)
from sklearn.feature_extraction import (
    DictVectorizer, 
    FeatureHasher
)
from collections import Counter
from imblearn.over_sampling import (
    RandomOverSampler, 
    SMOTE, 
    SVMSMOTE, 
    BorderlineSMOTE, 
    ADASYN, 
    SMOTEN, 
    SMOTENC
)
from imblearn.under_sampling import (
    RandomUnderSampler, 
    TomekLinks, 
    EditedNearestNeighbours, 
    RepeatedEditedNearestNeighbours, 
    AllKNN, 
    CondensedNearestNeighbour, 
    ClusterCentroids, 
    NearMiss
)
from sklearn.linear_model import BayesianRidge
from category_encoders import TargetEncoder, BinaryEncoder
from sklearn.feature_selection import (
    SelectKBest, 
    SelectFpr, 
    SelectFdr, 
    SelectFwe, 
    SelectPercentile, 
    GenericUnivariateSelect, 
    VarianceThreshold, 
    RFE, 
    RFECV, 
    SequentialFeatureSelector, 
    SelectFromModel, 
    f_regression, 
    chi2, 
    f_classif, 
    mutual_info_classif, 
    mutual_info_regression
)
from tpot import (
    TPOTClassifier, 
    TPOTRegressor
)
from sklearn.ensemble import (
    RandomForestClassifier, 
    RandomForestRegressor
)
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.decomposition import (
    PCA, 
    FactorAnalysis, 
    TruncatedSVD, 
    FastICA, 
    KernelPCA
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from tensorflow.keras.layers import Input, Dense # type: ignore
from tensorflow.keras.models import Model # type: ignore
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



def read_data(file_path: str, sheet_name: str = None, handle_duplicates: bool = True):
    """
    Read data from a file and return a DataFrame. Supports CSV, TXT, Excel, JSON, and HTML files.
    
    Parameters:
    - file_path: The path to the data file.
    - sheet_name: The name of the sheet to read from an Excel file (default is None).
    - handle_duplicates: Whether to drop duplicate rows (default is True).
    
    Returns:
    - A DataFrame or a list of DataFrames (in case of HTML).
    
    Raises:
    - ValueError: If the file format is not supported.
    """
    
    try:
        file_extension = file_path.split('.')[-1].lower()
        
        if file_extension in ['csv', 'txt']:
            data = pd.read_csv(file_path)
        elif file_extension == 'xlsx':
            if sheet_name is None:
                sheet_name = input('Enter the sheet name: ')
            data = pd.read_excel(file_path, sheet_name=sheet_name)
        elif file_extension == 'json':
            data = pd.read_json(file_path)
        elif file_extension == 'html':
            data = pd.read_html(file_path)
            if len(data) == 1:
                data = data[0]
        else:
            raise ValueError('Unsupported file format.')
        
        # Deep copy the data to avoid modifying the original data
        df = copy.deepcopy(data)
        
        # Handle duplicates if required
        if handle_duplicates:
            duplicated_num = df.duplicated().sum()
            if duplicated_num == 0:
                print('the DataFrame dont have any duplicates row')
            else:
                print(f'the DataFrame have {duplicated_num} duplicates rows')
                df = df.drop_duplicates()
                print('the DataFrame without duplicates rows')
        
        print(f'Data read successfully from {file_path}')
        return df
    
    except Exception as e:
        print(f'Error reading data from {file_path}: {str(e)}')
        raise


def columns_info(df):
    cols=[]
    dtype=[]
    unique_v=[]
    n_unique_v=[]
    
    for col in df.columns:
        cols.append(col)
        dtype.append(df[col].dtypes)
        unique_v.append(df[col].unique())
        n_unique_v.append(df[col].nunique())
    
    return pd.DataFrame({'names':cols,'dtypes':dtype,'unique':unique_v,'n_unique':n_unique_v}) 




def missing_rows(df, column_name=None):
    if column_name is not None and isinstance(column_name, str):
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' not found in DataFrame.")
        df.dropna(subset=[column_name], inplace=True)
    elif column_name is not None and isinstance(column_name, list):
        for col in column_name:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")
        df.dropna(subset=column_name, inplace=True)
    elif column_name is None:
        df.dropna(inplace=True)


def Missing_Values_DataFrame(df: pd.DataFrame, 
                         model: Literal['KNNImputer', 'SimpleImputer', 'IterativeImputer', 'constant', 'mean', 'median', 'mode', 'interpolation','Forward_fill','Backward_fill'] = 'KNNImputer', 
                         n_neighbors: int = 5, 
                         weights: str = 'uniform', 
                         strategy: str = 'mean', 
                         fill_value = None,
                         estimator = None,
                         max_iter = 10,
                         tol = 0.001,
                         constant: Union[int, float] = 0
                         ) -> pd.DataFrame:
    """
    Impute missing values in the DataFrame using the specified imputation strategy.
    
    Parameters:
    df (pd.DataFrame): DataFrame with missing values.
    model (str): Imputation strategy to use ('KNNImputer', 'SimpleImputer', 'constant', 
                'mean', 'median', 'mode', 'interpolation', 'IterativeImputer'). Default is 'KNNImputer'.
    n_neighbors (int): Number of neighbors to use for KNNImputer. Default is 5.
    weights (str): Weight function for KNNImputer. Default is 'uniform'.
    strategy (str): Strategy function for SimpleImputer. Default is 'mean'.
    constant (int/float): Value to fill missing values with when using 'constant'. Default is 0.
    
    Returns:
    pd.DataFrame: DataFrame with imputed values.
    
    Raises:
    ValueError: If an invalid model is specified.
    """
    # Identify columns with missing values
    missing_columns = df.columns[df.isnull().any()].tolist()
    print(f"Columns with missing values: {missing_columns}")
    
    valid_models = ['KNNImputer', 'SimpleImputer', 'IterativeImputer', 'constant', 'mean', 'median', 'mode', 'interpolation','Forward_fill','Backward_fill']
    if model not in valid_models:
        raise ValueError(f"Invalid model specified. Choose from {valid_models}")
    
    feature_columns = df.columns
    print(f'Starting imputation using {model} model')
    
    if model == 'KNNImputer':
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        df = imputer.fit_transform(df)
    elif model == 'SimpleImputer':
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy=strategy,fill_value=fill_value)
        df = imputer.fit_transform(df)
    elif model == 'IterativeImputer':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imputer = IterativeImputer(estimator=estimator, max_iter=max_iter, random_state=42, tol=tol)
        df = imputer.fit_transform(df)
    elif model == 'constant':
        df = df.fillna(constant)
    elif model == 'mean':
        df = df.fillna(df.mean())
    elif model == 'median':
        df = df.fillna(df.median())
    elif model == 'mode':
        df = df.apply(lambda x: x.fillna(x.mode()[0]), axis=0)
    elif model == 'interpolation':
        df = df.interpolate()
    elif model == 'Forward_fill':
        df = df.ffill()
    elif model == 'Backward_fill':
        df = df.bfill()
    
    
    df = pd.DataFrame(df, columns=feature_columns)
    
    print(f'Imputation completed using {model} model')
    return df


def Missing_Values_column(df: pd.DataFrame, 
                        columns: List[str], 
                        model: Literal['KNNImputer', 'SimpleImputer', 'IterativeImputer', 'constant', 'mean', 'median', 'mode', 'interpolation','Forward_fill','Backward_fill'] = 'KNNImputer', 
                        n_neighbors: int = 5, 
                        weights: str = 'uniform', 
                        strategy: str = 'mean', 
                        fill_value = None,
                        estimator = None,
                        max_iter = 10,
                        tol = 0.001,
                        constant: Union[int, float] = 0
                        ) -> pd.DataFrame:
    """
    Impute missing values in the specified columns of the DataFrame using the specified imputation strategy.
    
    Parameters:
    df (pd.DataFrame): DataFrame with missing values.
    columns (list of str): List of column names to impute.
    model (str): Imputation strategy to use ('KNNImputer', 'SimpleImputer', 'constant', 
                'mean', 'median', 'mode', 'interpolation', 'IterativeImputer'). Default is 'mean'.
    n_neighbors (int): Number of neighbors to use for KNNImputer. Default is 5.
    weights (str): Weight function for KNNImputer. Default is 'uniform'.
    strategy (str): Strategy function for SimpleImputer. Default is 'mean'.
    constant (int/float): Value to fill missing values with when using 'constant'. Default is 0.
    
    Returns:
    pd.DataFrame: DataFrame with imputed values.
    
    Raises:
    ValueError: If an invalid model is specified.
    """
    # Identify columns with missing values
    missing_columns = df.columns[df.isnull().any()].tolist()
    print(f"Columns with missing values: {missing_columns}")
    
    valid_models = ['KNNImputer', 'SimpleImputer', 'IterativeImputer', 'constant', 'mean', 'median', 'mode', 'interpolation','Forward_fill','Backward_fill']
    if model not in valid_models:
        raise ValueError(f"Invalid model specified. Choose from {valid_models}")
    
    print(f'Starting imputation for columns {columns} using {model} model')
    
    if model == 'KNNImputer':
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        df[columns] = imputer.fit_transform(df[columns])
    elif model == 'SimpleImputer':
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy=strategy,fill_value=fill_value)
        df[columns] = imputer.fit_transform(df[columns])
    elif model == 'IterativeImputer':
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imputer = IterativeImputer(estimator=estimator, max_iter=max_iter, random_state=42, tol=tol)
        df[columns] = imputer.fit_transform(df[columns])
    elif model == 'constant':
        df[columns] = df[columns].fillna(constant)
    elif model == 'mean':
        df[columns] = df[columns].fillna(df[columns].mean())
    elif model == 'median':
        df[columns] = df[columns].fillna(df[columns].median())
    elif model == 'mode':
        df[columns] = df[columns].apply(lambda x: x.fillna(x.mode()[0]), axis=0)
    elif model == 'interpolation':
        df[columns] = df[columns].interpolate()
    elif model == 'Forward_fill':
        df[columns] = df[columns].ffill()
    elif model == 'Backward_fill':
        df[columns] = df[columns].bfill()
    
    
    print(f'Imputation completed for columns {columns} using {model} model')
    return df


def check_outliers(df):
    outliers_df = df.drop(df.select_dtypes(exclude=['int64', 'float64']).columns.tolist(),axis=1)
    # Calculate the first and third quartiles of the data
    q1, q3 = np.percentile(outliers_df, [25, 75])

    # Calculate the IQR
    iqr = q3 - q1

    # Calculate the lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Identify the outliers
    outliers = (outliers_df < lower_bound) | (outliers_df > upper_bound)

    # Print the outliers
    print(outliers_df[outliers].count())

    return outliers_df


def box_plot_seaborn(
    data, x: Optional[str] = None, y: Optional[str] = None, hue: Optional[str] = None, 
    palette: str = 'magma',palette2: str = 'viridis',color=None, figsize: tuple = (8, 6), width: float = 0.5, whis: float = 1.5, 
    notch: bool = True, showmeans: bool = True, mean_marker: str = 'o', mean_color: str = 'black', 
    flier_marker: str = 'o', flier_size: int = 8, flier_color: str = 'black', flier_edge_color: str = 'purple', 
    xlabel: str = 'Groups', ylabel: str = 'Values', title: str = 'Box Plot (Seaborn)', 
    xtick_labels: Optional[List[str]] = None, font_scale: float = 1, orient: Optional[Literal['v', 'h', 'x', 'y']] = None
) -> None:
    """
    Create a box plot using Seaborn with the provided parameters.

    Parameters:
    - data: DataFrame
        The dataset for plotting.
    - x: str, optional
        Column name for x-axis variable.
    - y: str, optional
        Column name for y-axis variable.
    - hue: str, optional
        Column name for grouping variable to produce multiple plots.
    - palette: str
        Color palette for the plot.
    - figsize: tuple
        Size of the figure (width, height).
    - width: float
        Width of the box in the boxplot.
    - whis: float
        Whisker length in terms of IQR.
    - notch: bool
        Whether to draw a notch to indicate the confidence interval.
    - showmeans: bool
        Whether to show the mean value in the plot.
    - mean_marker: str
        Marker style for the mean value.
    - mean_color: str
        Color of the mean marker.
    - flier_marker: str
        Marker style for outliers.
    - flier_size: int
        Size of the outlier markers.
    - flier_color: str
        Color of the outlier markers.
    - flier_edge_color: str
        Edge color of the outlier markers.
    - xlabel: str
        Label for the x-axis.
    - ylabel: str
        Label for the y-axis.
    - title: str
        Title of the plot.
    - xtick_labels: list of str, optional
        Custom labels for the x-axis ticks.
    - font_scale: float
        Scaling factor for the font size of all text elements.
    - orient: {'v', 'h', 'x', 'y'}, optional
        Orientation of the plot (vertical or horizontal).
    
    Returns:
    - None
    """

    # Set font scale for all text elements and styling
    sns.set(font_scale=font_scale, style='white')

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # get the color for each plot
    if color is None:
        colors_list = random.sample(sns.color_palette(palette) + sns.color_palette(palette2), 12)
        colors = random.choices(colors_list, k=1)
    else:
        colors = color
    
    # Create the boxplot with the specified parameters
    if hue is not None:
        sns_plot = sns.boxplot(
            data=data, x=x, y=y, hue=hue, palette=palette, width=width, 
            whis=whis, notch=notch, showmeans=showmeans, orient=orient,
            meanprops=dict(marker=mean_marker, markerfacecolor=mean_color, markeredgecolor=mean_color),
            flierprops=dict(marker=flier_marker, markersize=flier_size, 
                            markerfacecolor=flier_color, markeredgecolor=flier_edge_color),
            ax=ax
        )
    elif hue is None and x is None and y is None:
        sns_plot = sns.boxplot(
            data=data, palette=palette, width=width, 
            whis=whis, notch=notch, showmeans=showmeans, orient=orient,
            meanprops=dict(marker=mean_marker, markerfacecolor=mean_color, markeredgecolor=mean_color),
            flierprops=dict(marker=flier_marker, markersize=flier_size, 
                            markerfacecolor=flier_color, markeredgecolor=flier_edge_color),
            ax=ax
        )
    elif hue is None:
        sns_plot = sns.boxplot(
            data=data, x=x, y=y, color=colors[0], width=width, 
            whis=whis, notch=notch, showmeans=showmeans, orient=orient,
            meanprops=dict(marker=mean_marker, markerfacecolor=mean_color, markeredgecolor=mean_color),
            flierprops=dict(marker=flier_marker, markersize=flier_size, 
                            markerfacecolor=flier_color, markeredgecolor=flier_edge_color),
            ax=ax
        )
    
    
    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=14 * font_scale)
    ax.set_ylabel(ylabel, fontsize=14 * font_scale)
    ax.set_title(title, fontsize=16 * font_scale)

    # Set custom x-axis tick labels if provided
    if xtick_labels:
        ax.set_xticks(range(len(xtick_labels)))
        ax.set_xticklabels(xtick_labels, fontsize=12 * font_scale)

    # Set y-axis tick font size
    ax.tick_params(axis='y', labelsize=12 * font_scale)

    # Add grid lines to the plot
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # Add a legend if a hue is used and there are labeled artists
    if hue and sns_plot.get_legend_handles_labels()[0]:
        ax.legend(title=hue, loc='upper right', fontsize=10 * font_scale)
    elif hue:
        print(f"Note: No legend created as no artists with labels were found for hue '{hue}'.")

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()


def scale_data(x_train: Union[np.ndarray, pd.DataFrame], 
               x_test: Union[np.ndarray, pd.DataFrame], 
               x_valid: Optional[Union[np.ndarray, pd.DataFrame]] = None, 
               scaler_type: Literal['standard', 'minmax', 'robust', 'maxabs', 'quantile', 'power', 'l2', 'log'] = 'standard') -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Scales the input data using the specified scaler type.
    
    Parameters:
    x_train (Union[np.ndarray, pd.DataFrame]): Training data.
    x_test (Union[np.ndarray, pd.DataFrame]): Test data.
    x_valid (Optional[Union[np.ndarray, pd.DataFrame]]): Validation data (optional).
    scaler_type (str): Type of scaler to use ('standard', 'minmax', 'robust', 'maxabs', 'quantile', 'power', 'l2', 'log'). Default is 'standard'.
    
    Returns:
    Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]: Returns the scaled (x_train, x_test) if x_valid is not provided,
    otherwise returns (x_train, x_valid, x_test).
    """
    
    def get_scaler(scaler_type: str, n_samples: int):
        """Returns the scaler object based on the scaler type."""
        if scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'minmax':
            return MinMaxScaler()
        elif scaler_type == 'robust':
            return RobustScaler()
        elif scaler_type == 'maxabs':
            return MaxAbsScaler()
        elif scaler_type == 'quantile':
            return QuantileTransformer(output_distribution='uniform', n_quantiles=min(n_samples, 1000))
        elif scaler_type == 'power':
            return PowerTransformer(method='yeo-johnson')
        elif scaler_type == 'l2':
            return Normalizer(norm='l2')
        elif scaler_type == 'log':
            return None  # Log transformation handled separately
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")

    def log_transform(*arrays):
        """Applies log transformation to the given arrays."""
        return tuple(np.log1p(array) for array in arrays)
    
    try:
        n_samples = x_train.shape[0]
        scaler = get_scaler(scaler_type, n_samples)
        
        if scaler_type == 'log':
            if x_valid is None:
                return log_transform(x_train, x_test)
            else:
                return log_transform(x_train, x_valid, x_test)
        
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        
        if x_valid is not None:
            x_valid_scaled = scaler.transform(x_valid)
            return x_train_scaled, x_test_scaled, x_valid_scaled
        else:
            return x_train_scaled, x_test_scaled
    
    except Exception as e:
        raise ValueError(f"An error occurred while scaling the data: {e}")


def encode_column(
    df: pd.DataFrame, 
    columns: Union[str, List[str]], 
    method: Literal[
        'get_dummies', 'label', 'ordinal', 'binary', 'target', 'dict_vectorizer', 
        'feature_hasher', 'label_binarizer', 'multi_label_binarizer', 'binary_encode'
    ] = 'get_dummies', 
    ordinal_categories: Optional[List[str]] = None, 
    target: Optional[str] = None, 
    n_features: Optional[int] = None,
    binary_default: bool = True
) -> pd.DataFrame:
    """
    Encodes one or more columns in the dataframe using the specified method.

    Parameters:
    ----------
    df (pd.DataFrame): The dataframe containing the column(s) to be encoded.
    columns (Union[str, List[str]]): The name of the column or a list of column names to be encoded.
    method (Literal['get_dummies', 'label', 'ordinal', 'binary', 'target', 
                    'dict_vectorizer', 'feature_hasher', 'label_binarizer', 
                    'multi_label_binarizer', 'binary_encode']): The encoding method to use. 
                    Options are 'get_dummies' (one_hot), 'label', 'ordinal', 
                    'binary', 'target', 'dict_vectorizer', 'feature_hasher', 
                    'label_binarizer', 'multi_label_binarizer', 'binary_encode'. Default is 'get_dummies'.
    ordinal_categories (Optional[List[str]]): Categories for ordinal encoding if method is 'ordinal'. Default is None.
    target (Optional[str]): Target column for target encoding. Default is None.
    n_features (Optional[int]): Number of features for feature hasher. Default is None.
    
    Examples:
    --------
    One-hot encoding for a single column
    >>> df_encoded = encode_column(df, 'column_name', method='get_dummies')
    Ordinal encoding with specified categories
    >>> df_encoded = encode_column(df, 'column_name', method='ordinal', ordinal_categories=['low', 'medium', 'high'])
    Binary encoding for a single column
    >>> df_encoded = encode_column(df, 'column_name', method='binary')
    Target encoding for a single column
    >>> df_encoded = encode_column(df, 'column_name', method='target', target='target_column')
    Feature hashing with a specified number of features
    >>> df_encoded = encode_column(df, 'column_name', method='feature_hasher', n_features=10)

    Returns:
    -------
    pd.DataFrame: The dataframe with the encoded column(s).

    """
    
    def binary(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Binary encodes a column with exactly two unique values, leaving NaN values unchanged."""
        unique_vals = df[column].dropna().unique()
        if len(unique_vals) != 2:
            raise ValueError("Column must have exactly two unique non-NaN values.")
        df[column] = df[column].apply(lambda x: 1 if x == unique_vals[0] else (0 if x == unique_vals[1] else np.nan))
        return df

    def binary_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Binary encodes a column using BinaryEncoder from category_encoders."""
        encoder = BinaryEncoder(cols=[column])
        df = encoder.fit_transform(df)
        df = df.infer_objects(copy=False)  # Ensures correct dtype inference without downcasting object dtype arrays
        return df

    def label_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Label encodes a column, preserving NaN values."""
        le = LabelEncoder()
        non_nan_mask = df[column].notna()
        le.fit(df.loc[non_nan_mask, column])
        df.loc[non_nan_mask, column] = le.transform(df.loc[non_nan_mask, column])
        return df

    def ordinal_encode(df: pd.DataFrame, column: str, categories: List[str]) -> pd.DataFrame:
        """Ordinal encodes a column with provided categories, preserving NaN values."""
        oe = OrdinalEncoder(categories=[categories], handle_unknown='use_encoded_value', unknown_value=np.nan)
        nan_mask = df[column].isna()
        df[column] = oe.fit_transform(df[[column]])
        df.loc[nan_mask, column] = np.nan
        return df

    def get_dummies(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """One-hot encodes a column, preserving NaN values."""
        nan_mask = df[column].isna()
        dummies = pd.get_dummies(df[column], drop_first=True, dtype=float)
        df = pd.concat([df, dummies], axis=1)
        df.loc[nan_mask, dummies.columns] = pd.NA
        df = df.drop(column, axis=1)
        return df

    def target_encode(df: pd.DataFrame, column: str, target: str) -> pd.DataFrame:
        """Target encodes a column based on the target column."""
        te = TargetEncoder(cols=[column])
        df[column] = te.fit_transform(df[column], df[target])
        return df

    def dict_vectorize(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Encodes a column using DictVectorizer."""
        dv = DictVectorizer(sparse=False)
        dict_data = df[column].apply(lambda x: {column: x})
        transformed = dv.fit_transform(dict_data)
        df = df.drop(column, axis=1)
        df = pd.concat([df, pd.DataFrame(transformed, columns=dv.get_feature_names_out())], axis=1)
        return df

    def feature_hash(df: pd.DataFrame, column: str, n_features: int) -> pd.DataFrame:
        """Encodes a column using FeatureHasher."""
        fh = FeatureHasher(n_features=n_features, input_type='string')
        transformed = fh.transform(df[column].astype(str)).toarray()
        df = df.drop(column, axis=1)
        df = pd.concat([df, pd.DataFrame(transformed, columns=[f'{column}_hashed_{i}' for i in range(n_features)])], axis=1)
        return df

    def label_binarize(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Label binarizes a column."""
        lb = LabelBinarizer()
        transformed = lb.fit_transform(df[column])
        df = df.drop(column, axis=1)
        df = pd.concat([df, pd.DataFrame(transformed, columns=lb.classes_)], axis=1)
        return df

    def multi_label_binarize(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Multi-label binarizes a column."""
        mlb = MultiLabelBinarizer()
        transformed = mlb.fit_transform(df[column])
        df = df.drop(column, axis=1)
        df = pd.concat([df, pd.DataFrame(transformed, columns=mlb.classes_)], axis=1)
        return df

    if isinstance(columns, str):
        columns = [columns]
    
    if binary_default:
        for column in columns:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' does not exist in the dataframe")
            
            unique_vals = df[column].value_counts()
            
            # If the column is binary (contains exactly two unique values)
            if len(unique_vals) == 2:
                df = binary(df, column)
            else:
                # Handle other encoding methods based on the method parameter
                if method == 'label':
                    df = label_encode(df, column)
                elif method == 'ordinal':
                    if ordinal_categories is None:
                        raise ValueError("ordinal_categories must be provided for ordinal encoding")
                    df = ordinal_encode(df, column, ordinal_categories)
                elif method == 'get_dummies':
                    df = get_dummies(df, column)
                elif method == 'target':
                    if target is None:
                        raise ValueError("Target column must be provided for target encoding")
                    df = target_encode(df, column, target)
                elif method == 'dict_vectorizer':
                    df = dict_vectorize(df, column)
                elif method == 'feature_hasher':
                    if n_features is None:
                        raise ValueError("Number of features must be provided for feature hasher")
                    df = feature_hash(df, column, n_features)
                elif method == 'label_binarizer':
                    df = label_binarize(df, column)
                elif method == 'multi_label_binarizer':
                    df = multi_label_binarize(df, column)
                elif method == 'binary_encode':
                    df = binary_encode(df, column)
                else:
                    raise ValueError(f"Encoding method '{method}' is not supported")
    else:
        for column in columns:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' does not exist in the dataframe")

            if method == 'binary':
                df = binary(df, column)
            elif method == 'label':
                df = label_encode(df, column)
            elif method == 'ordinal':
                if ordinal_categories is None:
                    raise ValueError("ordinal_categories must be provided for ordinal encoding")
                df = ordinal_encode(df, column, ordinal_categories)
            elif method == 'get_dummies':
                df = get_dummies(df, column)
            elif method == 'target':
                if target is None:
                    raise ValueError("Target column must be provided for target encoding")
                df = target_encode(df, column, target)
            elif method == 'dict_vectorizer':
                df = dict_vectorize(df, column)
            elif method == 'feature_hasher':
                if n_features is None:
                    raise ValueError("Number of features must be provided for feature hasher")
                df = feature_hash(df, column, n_features)
            elif method == 'label_binarizer':
                df = label_binarize(df, column)
            elif method == 'multi_label_binarizer':
                df = multi_label_binarize(df, column)
            elif method == 'binary_encode':
                df = binary_encode(df, column)
            else:
                raise ValueError(f"Encoding method '{method}' is not supported")
    return df


def calculate_correlation(df: pd.DataFrame, outcome_column: Optional[str] = None, num_results: Optional[int] = 5) -> pd.DataFrame:
    """
    Calculates and prints the Pearson correlation coefficient and p-value for each numeric column in the DataFrame
    against the specified outcome column, ordered by the Pearson correlation coefficient.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing data.
    outcome_column (str): The name of the outcome column to calculate the correlation against.
    num_results (int, optional): The number of top results to display. If None, display all results.

    Returns:
    pd.DataFrame: A DataFrame containing the Pearson correlation coefficients and p-values for each numeric column.
    """
    if outcome_column is None:
        outcome_column = df.iloc[:, -1].name
    
    if outcome_column not in df.columns:
        raise ValueError(f"The column '{outcome_column}' does not exist in the DataFrame.")
    
    # Select numeric columns from the DataFrame
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.to_list()
    
    # Ensure the outcome column is included in the numeric columns
    if outcome_column not in numeric_columns:
        raise ValueError(f"The outcome column '{outcome_column}' must be numeric and present in the DataFrame.")
    
    print('Calculating Pearson correlation coefficients for numeric columns against the outcome column: {outcome_column}')
    
    # Store the results
    results = []

    # Loop through each numeric column and calculate Pearson correlation
    for param in numeric_columns:
        if param != outcome_column:
            pearson_coef, p_value = stats.pearsonr(df[param].dropna(), df[outcome_column].dropna())
            results.append((param, pearson_coef, p_value))

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results, columns=['Variable', 'Pearson Coefficient', 'P-Value'])

    # Order the results by Pearson correlation coefficient
    results_df = results_df.reindex(results_df['Pearson Coefficient'].abs().sort_values(ascending=False).index)
    
    # Limit the number of results if num_results is specified
    if num_results is not None:
        results_df = results_df.head(num_results)
    
    print(f'Top {num_results if num_results is not None else len(results_df)} results:\n{results_df}')
    
    # Print the results
    for index, row in results_df.iterrows():
        print(f"{row['Variable']}")
        print(f"The Pearson Correlation Coefficient for {row['Variable']} is {row['Pearson Coefficient']:.4f} with a P-value of P = {row['P-Value']:.4g}\n")
    
    return results_df


def over_under_sampling_classification(
    x: pd.DataFrame, 
    y: pd.Series, 
    over_sampling: Literal['SMOTE', 'SVMSMOTE', 'BorderlineSMOTE-1', 'BorderlineSMOTE-2', 'ADASYN', 'SMOTEN', 'SMOTENC', 'random_over_sampler'] = 'SVMSMOTE', 
    under_sampling: Literal['TomekLinks', 'EditedNearestNeighbours', 'RepeatedEditedNearestNeighbours', 'AllKNN', 'CondensedNearestNeighbour', 'ClusterCentroids', 'NearMiss', 'random_under_sampler'] = 'TomekLinks', 
    over_sampling_strategy= "auto", 
    under_sampling_strategy= "auto", 
    k_neighbors: int = 5, 
    random_state: int = 42, 
    categorical_features: Optional[list] = None, 
    over: bool = True, 
    under: bool = True,
    make_df: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    print(f'Starting over-sampling and/or under-sampling process.')
    print(f'Initial class distribution: {Counter(y)}')
    
    valid_over_sampling_strategies = [
        'random_over_sampler', 'SMOTE', 'SVMSMOTE', 'BorderlineSMOTE-1', 
        'BorderlineSMOTE-2', 'ADASYN', 'SMOTEN', 'SMOTENC'
    ]
    valid_under_sampling_strategies = [
        'random_under_sampler', 'TomekLinks', 'EditedNearestNeighbours', 'RepeatedEditedNearestNeighbours', 
        'AllKNN', 'CondensedNearestNeighbour', 'ClusterCentroids', 'NearMiss'
    ]
    
    if over and over_sampling not in valid_over_sampling_strategies:
        raise ValueError(f"Invalid over_sampling strategy '{over_sampling}' specified. "
                        f"Valid options are: {', '.join(valid_over_sampling_strategies)}")
    
    if under and under_sampling not in valid_under_sampling_strategies:
        raise ValueError(f"Invalid under_sampling strategy '{under_sampling}' specified. "
                        f"Valid options are: {', '.join(valid_under_sampling_strategies)}")
    
    # Over-sampling
    if over:
        if over_sampling == 'SMOTE':
            print(f'Applying SMOTE with strategy {over_sampling_strategy}')
            smote = SMOTE(sampling_strategy=over_sampling_strategy, random_state=random_state, k_neighbors=k_neighbors)
            x, y = smote.fit_resample(x, y)
        elif over_sampling == 'SVMSMOTE':
            print(f'Applying SVMSMOTE with strategy {over_sampling_strategy}')
            svmsmote = SVMSMOTE(sampling_strategy=over_sampling_strategy, random_state=random_state)
            x, y = svmsmote.fit_resample(x, y)
        elif over_sampling == 'BorderlineSMOTE-1':
            print(f'Applying BorderlineSMOTE(kind="borderline-1") with strategy {over_sampling_strategy}')
            bl1smote = BorderlineSMOTE(kind='borderline-1', sampling_strategy=over_sampling_strategy, random_state=random_state)
            x, y = bl1smote.fit_resample(x, y)
        elif over_sampling == 'BorderlineSMOTE-2':
            print(f'Applying BorderlineSMOTE(kind="borderline-2") with strategy {over_sampling_strategy}')
            bl2smote = BorderlineSMOTE(kind='borderline-2', sampling_strategy=over_sampling_strategy, random_state=random_state)
            x, y = bl2smote.fit_resample(x, y)
        elif over_sampling == 'ADASYN':
            print(f'Applying ADASYN with strategy {over_sampling_strategy}')
            adasyn = ADASYN(sampling_strategy=over_sampling_strategy, random_state=random_state)
            x, y = adasyn.fit_resample(x, y)
        elif over_sampling == 'SMOTEN':
            print(f'Applying SMOTEN with strategy {over_sampling_strategy}')
            smoten = SMOTEN(sampling_strategy=over_sampling_strategy, random_state=random_state)
            x, y = smoten.fit_resample(x, y)
        elif over_sampling == 'SMOTENC':
            if categorical_features is None:
                raise ValueError("categorical_features must be provided for SMOTENC")
            print(f'Applying SMOTENC with strategy {over_sampling_strategy}')
            smotenc = SMOTENC(categorical_features=categorical_features, sampling_strategy=over_sampling_strategy, random_state=random_state)
            x, y = smotenc.fit_resample(x, y)
        elif over_sampling == 'random_over_sampler':
            print(f'Applying RandomOverSampler with strategy {over_sampling_strategy}')
            ros = RandomOverSampler(sampling_strategy=over_sampling_strategy, random_state=random_state)
            x, y = ros.fit_resample(x, y)
    
    # Under-sampling
    if under:
        if under_sampling == 'TomekLinks':
            print(f'Applying TomekLinks under-sampling.')
            tom = TomekLinks(n_jobs=-1)
            x, y = tom.fit_resample(x, y)
        elif under_sampling == 'EditedNearestNeighbours':
            print(f'Applying EditedNearestNeighbours with strategy {under_sampling_strategy}')
            enn = EditedNearestNeighbours(sampling_strategy=under_sampling_strategy, n_neighbors=3, kind_sel='all', n_jobs=-1)
            x, y = enn.fit_resample(x, y)
        elif under_sampling == 'RepeatedEditedNearestNeighbours':
            print(f'Applying RepeatedEditedNearestNeighbours with strategy {under_sampling_strategy}')
            renn = RepeatedEditedNearestNeighbours(sampling_strategy=under_sampling_strategy, n_neighbors=3, max_iter=100, kind_sel='all', n_jobs=-1)
            x, y = renn.fit_resample(x, y)
        elif under_sampling == 'AllKNN':
            print(f'Applying AllKNN with strategy {under_sampling_strategy}')
            allknn = AllKNN(sampling_strategy=under_sampling_strategy, n_neighbors=3, kind_sel='all', allow_minority=True, n_jobs=-1)
            x, y = allknn.fit_resample(x, y)
        elif under_sampling == 'CondensedNearestNeighbour':
            print(f'Applying CondensedNearestNeighbour with strategy {under_sampling_strategy}')
            cnn = CondensedNearestNeighbour(sampling_strategy=under_sampling_strategy, n_neighbors=1, random_state=random_state, n_jobs=-1)
            x, y = cnn.fit_resample(x, y)
        elif under_sampling == 'ClusterCentroids':
            print(f'Applying ClusterCentroids with strategy {under_sampling_strategy}')
            cc = ClusterCentroids(sampling_strategy=under_sampling_strategy, random_state=random_state, voting='soft')
            x, y = cc.fit_resample(x, y)
        elif under_sampling == 'NearMiss':
            print(f'Applying NearMiss(version=1) with strategy {under_sampling_strategy}')
            nm = NearMiss(sampling_strategy=under_sampling_strategy, version=1, n_neighbors=3, n_jobs=-1)
            x, y = nm.fit_resample(x, y)
        elif under_sampling == 'random_under_sampler':
            print(f'Applying RandomUnderSampler with strategy {under_sampling_strategy}')
            rus = RandomUnderSampler(sampling_strategy=under_sampling_strategy, random_state=random_state)
            x, y = rus.fit_resample(x, y)
    
    print(f'Final class distribution: {Counter(y)}')
    print(f'Over-sampling and/or under-sampling process completed.')
    if make_df:
        x_resampled = pd.DataFrame(x, columns=x.columns)
        y_resampled = pd.Series(y, name=y.name)
        combined_df = pd.concat([x_resampled, y_resampled], axis=1)
        return combined_df, x_resampled, y_resampled
    else:
        return x, y


def feature_selection(
    x_train: pd.DataFrame, 
    y_train: pd.Series, 
    x_test: pd.DataFrame,
    x_valid: Optional[pd.DataFrame] = None,
    method: Literal[
        'SelectKBest', 'SelectFpr', 'SelectFdr', 'SelectFwe', 'SelectPercentile', 
        'GenericUnivariateSelect', 'VarianceThreshold', 'RFE', 'RFECV', 
        'SequentialFeatureSelector', 'ExhaustiveFeatureSelector', 'SelectFromModel', 
        'TPOTClassifier', 'TPOTRegressor'] = 'SelectKBest', 
    stat_method: Optional[Literal[
        'f_regression', 'chi2', 'f_classif', 'mutual_info_classif', 'mutual_info_regression'
    ]] = 'f_regression', 
    k: int = 10, 
    percentile: int = 10, 
    alpha: float = 0.05, 
    threshold: float = 0.0, 
    n_features_to_select: Optional[int] = None, 
    cv: int = 5, 
    scoring: Optional[str] = None, 
    direction: Literal['forward', 'backward'] = 'forward', 
    estimator: Optional[Union[RandomForestClassifier, RandomForestRegressor]] = None, 
    generations: int = 5, 
    population_size: int = 50, 
    random_state: int = 42, 
    verbosity: int = 2,
    step = 1,
    n_jobs =-1,
    task: Literal['classification', 'regression'] = 'classification'
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Perform feature selection on the given dataset.

    Parameters:
    [... same as before ...]

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]: Transformed x_train, x_test, and optionally x_valid.
    """
    # Default estimator if none is provided
    if estimator is None:
        if task == 'classification':
            estimator = RandomForestClassifier(random_state=random_state)
        elif task == 'regression':
            estimator = RandomForestRegressor(random_state=random_state)
        else:
            raise ValueError("Invalid task. Choose 'classification' or 'regression'.")

    # Univariate feature selection methods
    stat_methods = {
        'f_regression': f_regression,
        'chi2': chi2,
        'f_classif': f_classif,
        'mutual_info_classif': mutual_info_classif,
        'mutual_info_regression': mutual_info_regression
    }
    
    if stat_method and stat_method not in stat_methods:
        raise ValueError(f"Invalid stat_method '{stat_method}' specified. "
                        f"Valid options are: {', '.join(stat_methods.keys())}")

    if method == 'SelectKBest':
        selector = SelectKBest(stat_methods[stat_method], k=k)
    elif method == 'SelectFpr':
        selector = SelectFpr(stat_methods[stat_method], alpha=alpha)
    elif method == 'SelectFdr':
        selector = SelectFdr(stat_methods[stat_method], alpha=alpha)
    elif method == 'SelectFwe':
        selector = SelectFwe(stat_methods[stat_method], alpha=alpha)
    elif method == 'SelectPercentile':
        selector = SelectPercentile(stat_methods[stat_method], percentile=percentile)
    elif method == 'GenericUnivariateSelect':
        selector = GenericUnivariateSelect(stat_methods[stat_method], mode='percentile', param=percentile)
    elif method == 'VarianceThreshold':
        selector = VarianceThreshold(threshold=threshold)
    elif method == 'RFE':
        selector = RFE(estimator, n_features_to_select=n_features_to_select)
    elif method == 'RFECV':
        selector = RFECV(estimator, step=step, cv=cv, scoring=scoring, n_jobs=n_jobs)
    elif method == 'SequentialFeatureSelector':
        selector = SequentialFeatureSelector(estimator, direction=direction, n_features_to_select=n_features_to_select, cv=cv, scoring=scoring, n_jobs=n_jobs)
    elif method == 'ExhaustiveFeatureSelector':
        selector = ExhaustiveFeatureSelector(estimator, min_features=1, max_features=n_features_to_select or x_train.shape[1], scoring=scoring, cv=cv, print_progress=True, n_jobs=n_jobs)
    elif method == 'SelectFromModel':
        selector = SelectFromModel(estimator)
    elif method == 'TPOTClassifier':
        if task != 'classification':
            raise ValueError("TPOTClassifier is only valid for classification tasks.")
        selector = TPOTClassifier(generations=generations, population_size=population_size, 
                                  cv=cv, random_state=random_state, verbosity=verbosity, n_jobs=n_jobs)
    elif method == 'TPOTRegressor':
        if task != 'regression':
            raise ValueError("TPOTRegressor is only valid for regression tasks.")
        selector = TPOTRegressor(generations=generations, population_size=population_size, 
                                 cv=cv, random_state=random_state, verbosity=verbosity, n_jobs=n_jobs)
    else:
        raise ValueError(f"Invalid method '{method}' specified. "
                        f"Valid options are: 'SelectKBest', 'SelectFpr', 'SelectFdr', 'SelectFwe', 'SelectPercentile', "
                        f"'GenericUnivariateSelect', 'VarianceThreshold', 'RFE', 'RFECV', 'SequentialFeatureSelector', "
                        f"'ExhaustiveFeatureSelector', 'SelectFromModel', 'TPOTClassifier', 'TPOTRegressor'.")

    # Fit selector to training data
    if method in ['TPOTClassifier', 'TPOTRegressor']:
        selector.fit(x_train, y_train)
        # Get the fitted pipeline
        fitted_pipeline = selector.fitted_pipeline_
        # Transform the data using the fitted pipeline
        x_train_new = fitted_pipeline.transform(x_train)
        x_test_new = fitted_pipeline.transform(x_test)
        x_valid_new = fitted_pipeline.transform(x_valid) if x_valid is not None else None
        
        # # Get feature names (this might vary depending on the pipeline steps)
        # feature_names = x_train.columns[fitted_pipeline.steps[-1][1].get_support()]
        
        # Convert to DataFrames with original feature names
        feature_names = x_train.columns[selector._fitted_imputer.feature_mask_]
        x_train_new = pd.DataFrame(x_train_new, columns=feature_names, index=x_train.index)
        x_test_new = pd.DataFrame(x_test_new, columns=feature_names, index=x_test.index)
        if x_valid is not None:
            x_valid_new = pd.DataFrame(x_valid_new, columns=feature_names, index=x_valid.index)
    elif method == 'ExhaustiveFeatureSelector':
        selector.fit(x_train, y_train, params={'sample_weight': None})
        x_train_new = selector.transform(x_train)
        x_test_new = selector.transform(x_test)
        x_valid_new = selector.transform(x_valid) if x_valid is not None else None
    else:
        selector.fit(x_train, y_train)
        x_train_new = pd.DataFrame(selector.transform(x_train), columns=x_train.columns[selector.get_support()])
        x_test_new = pd.DataFrame(selector.transform(x_test), columns=x_test.columns[selector.get_support()])
        x_valid_new = pd.DataFrame(selector.transform(x_valid), columns=x_valid.columns[selector.get_support()]) if x_valid is not None else None

    return x_train_new, x_test_new, x_valid_new


# Set the LOKY_MAX_CPU_COUNT environment variable
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Set to the number of physical cores you want to use


def dimensionality_reduction(
    x_train: pd.DataFrame, 
    x_test: pd.DataFrame,
    y_train: Optional[pd.Series] = None,
    method: Literal[
        'PCA', 'LDA', 'FactorAnalysis', 'TruncatedSVD', 'ICA', 
        'TSNE', 'UMAP', 'Autoencoder', 'KernelPCA'
    ] = 'PCA', 
    n_components: int = 10, 
    random_state: int = 42,
    perplexity: int = 30, 
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    kernel: Literal['linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'] = 'linear',
    autoencoder_hidden_layers: Optional[list] = None,
    x_valid: Optional[pd.DataFrame] = None,
    whiten: bool = False,
    svd_solver: Literal['auto', 'full', 'arpack', 'randomized'] = 'auto',
    solver: Literal['svd', 'lsqr', 'eigen'] = 'svd',
    shrinkage: Optional[Union[str, float]] = None,
    tol: float = 1e-2,
    algorithm: Literal['randomized', 'arpack'] = 'randomized',
    ica_algorithm: Literal['parallel', 'deflation'] = 'parallel',
    ica_whiten: Union[bool, Literal['arbitrary-variance', 'unit-variance']] = True,
    learning_rate: float = 200.0,
    max_iter: int = 1000,
    metric: str = 'euclidean',
    gamma: Optional[float] = None,
    degree: int = 3,
    coef0: float = 1.0,
    n_jobs: int = -1
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Union[PCA, LDA, FactorAnalysis, TruncatedSVD, FastICA, TSNE, umap.UMAP, KernelPCA, Model]]:
    if method == 'LDA':
        if y_train is None:
            raise ValueError("y_train must be provided for LDA.")
        n_features = x_train.shape[1]
        n_classes = y_train.nunique()
        max_components = min(n_features, n_classes - 1)
        if n_components > max_components:
            n_components = max_components
    
    if method == 'PCA':
        model = PCA(n_components=n_components, whiten=whiten, svd_solver=svd_solver, random_state=random_state)
    elif method == 'LDA':
        model = LDA(n_components=n_components, solver=solver, shrinkage=shrinkage)
    elif method == 'FactorAnalysis':
        model = FactorAnalysis(n_components=n_components, tol=tol, random_state=random_state)
    elif method == 'TruncatedSVD':
        model = TruncatedSVD(n_components=n_components, algorithm=algorithm, random_state=random_state)
    elif method == 'ICA':
        model = FastICA(n_components=n_components, algorithm=ica_algorithm, whiten=ica_whiten, random_state=random_state)
    elif method == 'TSNE':
        if n_components >= 4:
            raise ValueError("'n_components' should be less than 4 for the barnes_hut algorithm in TSNE.")
        model = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity, learning_rate=learning_rate, max_iter=max_iter, n_jobs=n_jobs)
        x_train_new = model.fit_transform(x_train)
        x_test_new = model.fit_transform(x_test)
        x_valid_new = model.fit_transform(x_valid) if x_valid is not None else None
        return pd.DataFrame(x_train_new), pd.DataFrame(x_test_new), pd.DataFrame(x_valid_new) if x_valid is not None else None, model
    elif method == 'UMAP':
        model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state, n_jobs=n_jobs)
    elif method == 'KernelPCA':
        model = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, random_state=random_state)
    elif method == 'Autoencoder':
        if autoencoder_hidden_layers is None:
            autoencoder_hidden_layers = [64, 32]

        input_dim = x_train.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoded = input_layer

        for units in autoencoder_hidden_layers:
            encoded = Dense(units, activation='relu')(encoded)

        encoded = Dense(n_components, activation='relu')(encoded)
        decoded = encoded

        for units in reversed(autoencoder_hidden_layers):
            decoded = Dense(units, activation='relu')(decoded)

        decoded = Dense(input_dim, activation='sigmoid')(decoded)

        autoencoder = Model(inputs=input_layer, outputs=decoded)
        encoder = Model(inputs=input_layer, outputs=encoded)

        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_valid, x_valid) if x_valid is not None else None, verbose=0)
        
        x_train_new = encoder.predict(x_train)
        x_test_new = encoder.predict(x_test)
        x_valid_new = encoder.predict(x_valid) if x_valid is not None else None
        return pd.DataFrame(x_train_new), pd.DataFrame(x_test_new), pd.DataFrame(x_valid_new) if x_valid is not None else None, autoencoder
    else:
        raise ValueError(f"Invalid method '{method}' specified. "
                         f"Valid options are: 'PCA', 'LDA', 'FactorAnalysis', 'TruncatedSVD', 'ICA', "
                         f"'TSNE', 'UMAP', 'Autoencoder', 'KernelPCA'.")

    if method == 'LDA':
        model.fit(x_train, y_train)  # LDA requires y_train for fitting
    else:
        model.fit(x_train)

    x_train_new = model.transform(x_train)
    x_test_new = model.transform(x_test)
    x_valid_new = model.transform(x_valid) if x_valid is not None else None

    return pd.DataFrame(x_train_new), pd.DataFrame(x_test_new), pd.DataFrame(x_valid_new) if x_valid is not None else None, model















































































































