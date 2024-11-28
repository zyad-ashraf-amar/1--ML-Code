import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from pandas.plotting import parallel_coordinates
import copy



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
    number_of_rows = df.shape[0]
    number_of_null = []
    for col in df.columns:
        cols.append(col)
        dtype.append(df[col].dtypes)
        unique_v.append(df[col].unique())
        n_unique_v.append(df[col].nunique())
        number_of_null.append(df[col].isnull().sum())
    
    return pd.DataFrame({'names':cols, 'dtypes':dtype, 'unique':unique_v, 'n_unique':n_unique_v, 'number_of_rows':number_of_rows, 'number_of_null':number_of_null})



def bar_plot(df,col):
    
    fig = px.bar(df,
        x = df[col].value_counts().keys(), 
        y = df[col].value_counts().values,
        color= df[col].value_counts().keys())
    fig.update_layout(
    xaxis_title= col,
    yaxis_title="Count",
    legend_title=col,
    font_family="Courier New",
    font_color="blue",
    title_font_family="Times New Roman",
    title_font_color="red",
    legend_title_font_color="green")
    fig.show()


def transaction_to_df(transactions):
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    te_df = pd.DataFrame(te_ary, columns = te.columns_)
    return te_df


def association_rules_apriori(te_df, min_support = 0.001, min_threshold = 0.001, metric = 'lift', use_colnames = True, verbose = 1):
    freq_items = apriori(te_df, min_support = min_support, use_colnames = use_colnames, verbose = verbose)
    freq_items['length'] = freq_items['itemsets'].apply(lambda x: len(x))
    rules =  association_rules(freq_items,metric = metric,min_threshold = min_threshold)
    return freq_items, rules


def plot_rule_px_scatter(rules, x = 'support', y = 'confidence'):
    fig=px.scatter(rules[x], rules[y])
    fig.update_layout(
        xaxis_title="support",
        yaxis_title="confidence",
    
        font_family="Courier New",
        font_color="black",
        title_font_family="Times New Roman",
        title_font_color="red",
        title=(f'{x} vs {y}')
        
    )
    fig.show()



def plot_rule_scatter(rules, x = 'support', y = 'confidence'):
    plt.figure(figsize=(10, 6))
    plt.scatter(rules[x], rules[y], alpha=0.8)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{x} vs {y} Scatter Plot')
    plt.tight_layout()
    plt.show()



def plot_rule_polyfit(rules, x = 'lift', y = 'confidence'):
    fit = np.polyfit(rules[x], rules[y], 1)
    fit_fn = np.poly1d(fit)
    plt.plot(rules[x], rules[y], 'yo', rules[x], 
    fit_fn(rules[x]))
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'{x} vs {y}')




























