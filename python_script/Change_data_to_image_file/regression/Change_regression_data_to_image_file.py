import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
from PIL import Image

# Create a dictionary with the table data
data = {
    "Technique": [
        "Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net Regression", "Polynomial Regression",
        "KNeighbors Regressor (KNN Regressor)", "Support Vector Machine (SVR)", "Decision Tree Regression", 
        "Bagging Regression", "Random Forest Regression", "Extra Trees Regression", "AdaBoost Regression", 
        "Gradient Boosting Regression", "CatBoost Regression", "XGBoost Regression", "Stacking Regressor", 
        "Bayesian Regression", "Multi-layer Perceptron (MLP) Regressor", "Convolutional Neural Networks (CNN) for Regression", 
        "Recurrent Neural Networks (RNN) for Regression", "Stochastic Gradient Descent (SGD) Regressor", 
        "Principal Component Regression (PCR)", "Partial Least Squares Regression (PLSR)", "Quantile Regression", 
        "Tweedie Regression", "Generalized Linear Models (GLM)", "Poisson Regression", "Gamma Regression", "Binomial Regression"
    ],
    "Coding Complexity": [
        "Low", "Medium", "Medium", "Medium", "Medium", "Low", "High", "Medium", "High", "High", "High", "High", "High", "High", "High", "High", "High",
        "High", "High", "High", "Medium", "Medium", "Medium", "Medium", "Medium", "Medium", "Medium", "Medium", "Medium"
    ],
    "Typical Use Cases": [
        "Simple linear relationships", "General regression tasks", "Feature selection, general regression", "General regression tasks", "Non-linear relationships", 
        "Non-parametric regression, local prediction", "High-dimensional data, complex relationships", "Interpretable models, non-linear relationships", 
        "Reducing variance, improving stability", "General regression tasks, handling large datasets", "Reducing variance, general regression tasks", 
        "Improving model performance, handling bias", "General regression tasks, handling complex datasets", "Categorical features, handling large datasets", 
        "General regression tasks, handling large datasets", "Combining multiple regression models", "Probabilistic regression, uncertainty estimation", 
        "Complex non-linear relationships, high-dimensional data", "Image data, spatial relationships", "Sequential data, time series analysis", 
        "Large-scale datasets, online learning", "Dimensionality reduction, multicollinearity", "Multicollinearity, high-dimensional data", 
        "Predicting quantiles, heteroscedasticity", "Generalized linear models, handling non-normal distributions", "Various types of regression (logistic, Poisson, etc.)", 
        "Count data, rate data", "Continuous positive data, rate data", "Binary response data"
    ],
    "When to Use": [
        "When relationship between variables is linear", "When multicollinearity is present", "When feature selection is important", "When there are many correlated features", 
        "When data shows polynomial relationships", "When there are non-linear relationships", "When data is high-dimensional or complex", "When interpretability is important", 
        "When variance reduction is needed", "When robustness and accuracy are important", "When variance reduction is needed", "When dealing with weak learners", 
        "When handling complex datasets", "When dealing with categorical features", "When robustness and accuracy are important", "When leveraging multiple models' strengths", 
        "When uncertainty estimation is important", "When data is high-dimensional or complex", "When working with image or spatial data", "When dealing with sequential or time-series data", 
        "When dealing with large-scale data", "When dealing with high-dimensional data", "When dealing with high-dimensional data", "When dealing with heteroscedastic data", 
        "When dealing with non-normal distributions", "When dealing with non-normal distributions", "When dealing with count or rate data", "When dealing with positive continuous data", 
        "When dealing with binary outcome data"
    ],
    "Good with Overfitting": [
        "No", "Yes", "Yes", "Yes", "No", "No", "Yes", "No", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", 
        "No", "No", "No", "Yes", "Yes", "Yes", "Yes", "Yes"
    ],
    "Handles Outliers": [
        "No", "No", "No", "No", "No", "No", "Yes (with appropriate kernel and parameters)", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", 
        "Yes (depending on base learners)", "No", "No", "No", "No", "No", "No", "No", "Yes", "No", "No", "No", "No", "No"
    ],
    "Strengths": [
        "Easy to understand, quick to implement", "Reduces overfitting, handles multicollinearity", "Can set coefficients to zero, useful for feature selection", 
        "Combines L1 and L2 regularization, robust to collinearity", "Captures non-linear patterns", "Simple, no assumptions about data distribution", 
        "Effective in high dimensions, robust to overfitting", "Easy to interpret, captures non-linear relationships", "Reduces variance, improves stability", 
        "Handles large datasets well, reduces overfitting", "Reduces variance, faster than Random Forest", "Boosts weak learners, reduces bias", 
        "Reduces bias and variance, handles complex data", "Handles categorical features automatically, reduces overfitting", "Highly efficient, reduces overfitting", 
        "Combines multiple models, improves performance", "Provides probabilistic predictions, handles overfitting", "Captures complex relationships, flexible architecture", 
        "Captures spatial hierarchies, effective for image data", "Captures temporal dependencies, effective for time series", "Efficient for large datasets, online learning", 
        "Reduces dimensionality, handles multicollinearity", "Reduces dimensionality, handles multicollinearity", "Provides quantile predictions, robust to outliers", 
        "Flexible, handles various distributions", "Flexible, handles various distributions", "Handles count data, interpretable", "Handles skewed distributions, interpretable", 
        "Handles binary data, interpretable"
    ],
    "Weaknesses": [
        "Prone to overfitting, not suitable for complex relationships", "Requires tuning of regularization parameter", "Can eliminate important features", 
        "Requires tuning of two regularization parameters", "Prone to overfitting with high-degree polynomials", "Computationally expensive, sensitive to outliers", 
        "Computationally intensive, requires parameter tuning", "Prone to overfitting, can be unstable", "Computationally expensive, less interpretable", 
        "Computationally intensive, less interpretable", "Less interpretable", "Sensitive to noisy data", "Computationally expensive, sensitive to overfitting", 
        "Computationally intensive, requires parameter tuning", "Computationally intensive, requires parameter tuning", "Computationally intensive, complex to implement", 
        "Computationally intensive, complex to implement", "Computationally intensive, requires large datasets", "Requires large datasets, computationally intensive", 
        "Computationally intensive, requires large datasets", "Requires careful tuning of learning rate", "Can lose interpretability, sensitive to scaling", 
        "Can lose interpretability", "Computationally intensive", "Requires careful selection of parameters", "Requires careful selection of link function", 
        "Assumes mean equals variance, not suitable for over-dispersed data", "Requires careful selection of link function", "Requires careful selection of link function"
    ],
    "Other Information": [
        "", "Uses L2 regularization", "Uses L1 regularization", "", "Degree of polynomial needs to be chosen carefully", "Number of neighbors (k) needs to be chosen", 
        "Uses kernel trick to handle non-linear data", "Requires pruning to avoid overfitting", "Combines multiple models to improve performance", "Ensemble of decision trees", 
        "Uses random splits", "Combines weak learners sequentially", "Sequential ensemble of weak learners", "Gradient boosting technique", "Gradient boosting technique", 
        "Requires careful selection of base models", "Based on Bayesian inference", "Neural network-based technique", "Neural network-based technique", "Neural network-based technique", 
        "Uses stochastic gradient descent for optimization", "Combines PCA with linear regression", "Similar to PCR but considers response variable", "Predicts conditional quantiles", 
        "Family of exponential dispersion models", "Includes various types of regression models", "Special case of GLM", "Special case of GLM", "Special case of GLM"
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Export the DataFrame to an Excel file
excel_path = 'regression_techniques.xlsx'
df.to_excel(excel_path, index=False)

print(f"The table has been exported to {excel_path}")

# # Export the DataFrame to an csv file
# csv_path = 'regression_techniques.csv'
# df.to_csv(excel_path, index=False)

# print(f"The table has been exported to {csv_path}")

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(24, 13))  # Set the figure size
ax.axis('tight')
ax.axis('off')

# Create the table plot with custom column widths
colWidths = [0.11, 0.045, 0.12, 0.1, 0.048, 0.09, 0.125, 0.14, 0.12]  # Adjust the widths as needed
tbl = table(ax, df, loc='center', cellLoc='center', colWidths=colWidths)

# Style the table
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(2, 2)  # Adjust the scaling to fit the table

# Set colors for the table
header_color = '#40466e'
row_colors = ['#f1f1f2', 'w']
edge_color = 'w'

for k, cell in tbl.get_celld().items():
    cell.set_edgecolor(edge_color)
    if k[0] == 0:
        cell.set_text_props(weight='bold', color='w')
        cell.set_facecolor(header_color)
    else:
        cell.set_facecolor(row_colors[k[0] % len(row_colors)])
        cell.set_text_props(color='black')

# Save the table as an image
plt.savefig('regression_techniques.png', bbox_inches='tight', dpi=300)

# Open the image using PIL to ensure it's saved correctly
img = Image.open('regression_techniques.png')
img.show()



data = {
    "Technique": [
        "Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net Regression", "Polynomial Regression",
        "KNeighbors Regressor (KNN Regressor)", "Support Vector Machine (SVR)", "Decision Tree Regression", 
        "Bagging Regression", "Random Forest Regression", "Extra Trees Regression", "AdaBoost Regression", 
        "Gradient Boosting Regression", "CatBoost Regression", "XGBoost Regression", "StackingRegressor", 
        "Bayesian Regression", "Multi-layer Perceptron (MLP) Regressor", "Convolutional Neural Networks (CNN) for Regression", 
        "Recurrent Neural Networks (RNN) for Regression", "Stochastic Gradient Descent (SGD) Regressor", 
        "Principal Component Regression (PCR)", "Partial Least Squares Regression (PLSR)", "Quantile Regression", 
        "Tweedie Regression", "Generalized Linear Models (GLM)", "Poisson Regression", "Gamma Regression", "Binomial Regression"
    ],
    "Coding Complexity": [
        "Low", "Low-Medium", "Low-Medium", "Medium", "Medium", "Low", "Medium", "Low", "Medium", "Medium", "Medium", "Medium",
        "Medium-High", "Medium-High", "Medium-High", "High", "Medium", "High", "Very High", "Very High", "Low", "Medium",
        "Medium", "Medium", "Medium", "Medium", "Medium", "Medium", "Medium"
    ],
    "Typical Use Cases": [
        "Simple predictive modeling, relationship analysis", "When multicollinearity is present", "Feature selection, sparse models",
        "Combines Lasso and Ridge benefits", "Non-linear relationships", "Pattern recognition, anomaly detection",
        "High-dimensional spaces, non-linear relationships", "Hierarchical decision making, interpretable models",
        "Reducing variance, ensemble method", "General-purpose regression, feature importance", "Similar to Random Forest, but faster",
        "Boosting weak learners", "High performance regression", "Handling categorical features", "High performance, large datasets",
        "Combining multiple models", "Uncertainty quantification", "Complex non-linear relationships", "Image-based regression tasks",
        "Time series, sequential data", "Large scale learning", "High-dimensional data, multicollinearity",
        "High-dimensional data, multicollinearity", "Predicting specific quantiles", "Modeling non-negative data with exact zeros",
        "Various response distributions", "Count data", "Modeling positive continuous data", "Binary outcomes, proportions"
    ],
    "When to Use": [
        "Linear relationships, baseline model", "Many features, correlated predictors", "Many irrelevant features",
        "Many features, some correlation", "Curvilinear patterns in data", "Non-linear relationships, sufficient data",
        "Complex patterns, outlier resistance needed", "Non-linear relationships, feature interactions",
        "High variance in base models", "Complex relationships, feature selection needed",
        "Need for faster training than Random Forest", "Improving simple models", "Need for high accuracy",
        "Many categorical variables", "Need for speed and performance", "Leveraging strengths of different models",
        "Need for probabilistic predictions", "Large datasets, complex patterns", "Spatial data (e.g., images)",
        "Temporal dependencies", "Very large datasets", "Many correlated features",
        "Many correlated features, multiple outputs", "Need for specific quantile predictions",
        "Insurance claims, rainfall data", "Non-normal error distributions", "Modeling event counts or rates",
        "Skewed positive data (e.g., insurance claims)", "Modeling probabilities or proportions"
    ],
    "Good with Overfitting": [
        "No", "Yes", "Yes", "Yes", "No", "No", "Yes", "No", "Yes", "Yes", "Yes", "Depends", "Depends", "Yes", "Yes",
        "Depends", "Yes", "Depends", "Depends", "Depends", "No", "Yes", "Yes", "No", "No", "No", "No", "No", "No"
    ],
    "Handle the outliers": [
        "Yes",  # Linear_regression
        "Yes",  # Ridge_Regression
        "Yes",  # Lasso_Regression
        "Yes",  # Elastic_Net_Regression
        "Yes",  # Polynomial_Regression
        "No", # KNeighbors_Regressor_(KNN_Regressor)
        "Yes",  # Support_vector_machine_(SVR)
        "No", # Decision_Tree_Regression
        "No", # Bagging_Regression
        "No", # Random_Forest_Regression
        "No", # Extra_Trees_Regression
        "No", # AdaBoost_Regression
        "No", # Gradient_Boosting_Regression
        "No", # CatBoost_Regression
        "No", # XGBoost_Regression
        "No", # StackingRegressor
        "Yes",  # Bayesian Regression
        "Yes",  # Multi-layer Perceptron (MLP) Regressor
        "Yes",  # Convolutional Neural Networks (CNN) for Regression
        "Yes",  # Recurrent Neural Networks (RNN) for Regression
        "Yes",  # Stochastic Gradient Descent (SGD) Regressor
        "Yes",  # Principal Component Regression (PCR)
        "Yes",  # Partial Least Squares Regression (PLSR)
        "No", # Quantile Regression
        "Yes",  # Tweedie Regression
        "Yes",  # Generalized Linear Models (GLM)
        "Yes",  # Poisson Regression
        "Yes",  # Gamma Regression
        "Yes"   # Binomial Regression
        ],
        "need scaling data": [
        "Yes",  # Linear_regression
        "Yes",  # Ridge_Regression
        "Yes",  # Lasso_Regression
        "Yes",  # Elastic_Net_Regression
        "Yes",  # Polynomial_Regression
        "Yes",  # KNeighbors_Regressor_(KNN_Regressor)
        "Yes",  # Support_vector_machine_(SVR)
        "No", # Decision_Tree_Regression
        "No", # Bagging_Regression
        "No", # Random_Forest_Regression
        "No", # Extra_Trees_Regression
        "No", # AdaBoost_Regression
        "No", # Gradient_Boosting_Regression
        "No", # CatBoost_Regression
        "No", # XGBoost_Regression
        "Yes",  # StackingRegressor
        "Yes",  # Bayesian Regression
        "Yes",  # Multi-layer Perceptron (MLP) Regressor
        "Yes",  # Convolutional Neural Networks (CNN) for Regression
        "Yes",  # Recurrent Neural Networks (RNN) for Regression
        "Yes",  # Stochastic Gradient Descent (SGD) Regressor
        "Yes",  # Principal Component Regression (PCR)
        "Yes",  # Partial Least Squares Regression (PLSR)
        "Yes",  # Quantile Regression
        "Yes",  # Tweedie Regression
        "Yes",  # Generalized Linear Models (GLM)
        "Yes",  # Poisson Regression
        "Yes",  # Gamma Regression
        "Yes"   # Binomial Regression
        ],
    "Strengths": [
        "Simple, interpretable, fast", "Handles multicollinearity, prevents overfitting", "Feature selection, sparse solutions",
        "Balances L1 and L2 penalties", "Can model complex relationships", "Non-parametric, handles complex patterns",
        "Handles non-linearity well, robust to outliers", "Interpretable, handles feature interactions",
        "Reduces overfitting, handles complex relationships", "Handles non-linearity, feature importance",
        "Faster than Random Forest, handles non-linearity", "Can improve weak models significantly",
        "High accuracy, handles various data types", "Handles categorical features well, fast",
        "Fast, high performance, regularization", "Can achieve higher accuracy", "Provides uncertainty estimates",
        "Can model very complex relationships", "Excellent for spatial data", "Handles sequential data well",
        "Efficient for large datasets", "Reduces dimensionality", "Handles multicollinearity, multiple outputs",
        "Robust to outliers, models full conditional distribution", "Handles mixed discrete-continuous distributions",
        "Flexible for different types of response variables", "Appropriate for count data",
        "Handles right-skewed data well", "Appropriate for binary data"
    ],
    "Weaknesses": [
        "Assumes linearity, sensitive to outliers", "May underestimate coefficients", "May be unstable with correlated features",
        "Requires tuning of two hyperparameters", "Prone to overfitting, sensitive to outliers",
        "Sensitive to feature scaling, slow for large datasets", "Can be slow on large datasets, requires feature scaling",
        "Prone to overfitting, unstable", "Can be computationally expensive, less interpretable",
        "Black box model, can be computationally expensive", "Slightly lower performance than Random Forest",
        "Sensitive to noisy data and outliers", "Can overfit if not tuned properly, slower training",
        "Newer, less community support", "Complex to tune, can overfit",
        "Complex, can overfit, computationally expensive", "Can be computationally expensive",
        "Requires large datasets, black box model", "Requires very large datasets, computationally intensive",
        "Complex to train, prone to vanishing gradients", "Requires feature scaling, sensitive to hyperparameters",
        "May lose important information", "Can be difficult to interpret", "Can be computationally intensive",
        "Can be complex to interpret", "Assumes linear relationship with link function",
        "Assumes equal mean and variance", "Limited to positive continuous data",
        "Limited to binary outcomes or proportions"
    ],
    "Additional Information": [
        "Ordinary Least Squares (OLS) method", "Also known as L2 regularization", "Also known as L1 regularization",
        "Good for high-dimensional data", "Degree of polynomial is a hyperparameter", "K is an important hyperparameter",
        "Kernel choice is crucial", "Tree depth is a key hyperparameter", "Often uses decision trees as base estimators",
        "Number of trees is a key hyperparameter", "Random split points for feature selection",
        "Adjusts subsequent classifiers to misclassified instances", "Builds trees sequentially", "Uses ordered boosting",
        "Popular in competitions", "Requires careful selection of base models", "Uses prior probabilities",
        "Neural network-based approach", "Often used in computer vision tasks", "Variants include LSTM and GRU",
        "Online learning capable", "Combines PCA with linear regression",
        "Similar to PCR but uses dependent variable", "Useful for risk assessment",
        "Part of the exponential dispersion family", "Extends linear regression to other distributions",
        "Used for rare event prediction", "Often used in actuarial science", "Logistic regression is a special case"
    ]
}





