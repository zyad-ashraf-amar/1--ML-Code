import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
from PIL import Image

# Create a dictionary with the table data
data = {
    "Technique": [
        "Logistic Regression", "GaussianNB", "MultinomialNB", "BernoulliNB", 
        "K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)", "Decision Tree (DT)", 
        "Random Forest Classifier", "Gradient Boosting Classifier", "LightGBM", "CatBoost Classifier", 
        "XGBoost Classifier", "Extra Trees Classifier", "Bagging Classifier", "AdaBoost Classifier", 
        "StackingClassifier", "Neural Networks", "Multi-layer Perceptron (MLP) Classifier", 
        "Convolutional Neural Networks (CNN) Classifier", "Recurrent Neural Networks (RNN) Classifier", 
        "Stochastic Gradient Descent (SGD) Classifier", "Linear Discriminant Analysis (LDA) Classifier", 
        "Quadratic Discriminant Analysis (QDA) Classifier", "Extreme Learning Machines (ELM) Classifier", 
        "Regularized Discriminant Analysis (RDA) Classifier"
    ],
    "Coding Complexity": [
        "Low", "Low", "Low", "Low", "Medium", "High", "Medium", "High", "High", "High", "High", 
        "High", "High", "Medium", "High", "High", "Very High", "High", "Very High", "Very High", 
        "Medium", "Medium", "Medium", "High", "High"
    ],
    "Typical Use Cases": [
        "Binary classification, medical diagnosis", "Text classification, spam detection", 
        "Text classification, spam detection", "Text classification with binary features", 
        "Image recognition, recommendation systems", "Text classification, image recognition", 
        "Customer segmentation, credit risk analysis", "Credit scoring, fraud detection", 
        "Web search ranking, customer churn prediction", "Large-scale machine learning tasks, real-time prediction", 
        "E-commerce, recommendation systems", "Web search ranking, click-through rate prediction", 
        "Various machine learning tasks", "Ensemble learning tasks", "Face detection, text classification", 
        "Ensemble learning tasks", "Image and speech recognition, natural language processing", 
        "Pattern recognition, time series forecasting", "Image and video recognition, medical image analysis", 
        "Sequence prediction, language modeling", "Large-scale and sparse machine learning problems", 
        "Face recognition, customer classification", "Face recognition, medical diagnosis", 
        "Real-time learning tasks", "Medical diagnosis, biometric recognition"
    ],
    "When to Use": [
        "Simple binary classification problems", "When features are continuous and normally distributed", 
        "When features are counts or frequencies", "When features are binary", 
        "When simplicity and interpretability are required", "When classes are well-separated and non-linear boundaries", 
        "When interpretability is important", "When accuracy and robustness are important", 
        "When high predictive accuracy is required", "When dealing with large datasets", 
        "When dealing with categorical data", "When high predictive accuracy is required", 
        "When reducing overfitting and variance is important", "When improving stability and accuracy of models", 
        "When improving weak learners", "When combining multiple classifiers to improve performance", 
        "When capturing complex patterns in large datasets", "When dealing with non-linear and complex relationships", 
        "When dealing with grid-like data structures (images, videos)", "When dealing with sequential data", 
        "When dealing with large datasets and sparse data", "When dealing with linearly separable data", 
        "When dealing with linearly and non-linearly separable data", "When fast training is required", 
        "When dealing with small sample sizes"
    ],
    "Good with Overfitting": [
        "No", "No", "No", "No", "No", "Yes", "Yes", "No", "No", "No", "No", "No", 
        "No", "No", "No", "No", "Yes", "Yes", "Yes", "Yes", "No", "No", "No", "No", "No"
    ],
    "Good with Unbalanced Data": [
        "Yes", "Yes", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "No", "Yes", "Yes", "Yes", 
        "Yes", "Yes", "No", "Yes", "No", "No", "No", "No", "Yes", "Yes", "Yes", "No", "Yes"
    ],
    "Handle the outliers": [
        "Sensitive to outliers; standardize data or use robust scalers.",
        "Not sensitive to outliers; uses Gaussian distribution.",
        "Not sensitive to outliers; suitable for discrete data.",
        "Not sensitive to outliers; suitable for binary data.",
        "Sensitive to outliers; consider removing outliers or use robust distance metrics.",
        "Sensitive to outliers; use robust kernels or outlier detection techniques.",
        "Insensitive to outliers; tree-based methods are robust.",
        "Insensitive to outliers; tree-based methods are robust.",
        "Moderately sensitive to outliers; consider removing outliers.",
        "Moderately sensitive to outliers; consider removing outliers.",
        "Moderately sensitive to outliers; consider removing outliers.",
        "Moderately sensitive to outliers; consider removing outliers.",
        "Insensitive to outliers; tree-based methods are robust.",
        "Insensitive to outliers; tree-based methods are robust.",
        "Sensitive to outliers; outliers can be given too much weight.",
        "Depends on the base classifiers used.",
        "Sensitive to outliers; normalize data or use robust loss functions.",
        "Sensitive to outliers; normalize data or use robust loss functions.",
        "Sensitive to outliers; data augmentation can help.",
        "Sensitive to outliers; normalize data or use robust loss functions.",
        "Sensitive to outliers; standardize data or use robust loss functions.",
        "Sensitive to outliers; consider removing or transforming outliers.",
        "Sensitive to outliers; consider removing or transforming outliers.",
        "Sensitive to outliers; standardize data or use robust loss functions.",
        "Sensitive to outliers; consider removing or transforming outliers."
    ],
    "Strengths": [
        "Interpretable, fast to train, well-calibrated probabilities", 
        "Fast, works well with small datasets, handles missing data", 
        "Fast, handles large feature spaces well", 
        "Fast, good for binary/boolean features", 
        "Simple to understand, effective in low-dimensional spaces", 
        "Effective in high-dimensional spaces, can handle non-linear boundaries", 
        "Easy to interpret, handles categorical data well", 
        "Reduces overfitting by averaging multiple trees, robust to noise", 
        "High accuracy, works well with a variety of loss functions", 
        "Fast training, handles large datasets and high-dimensional data well", 
        "Handles categorical data well, robust to overfitting, less hyperparameter tuning required", 
        "High performance, effective with sparse data, robust to overfitting", 
        "Reduces overfitting by averaging multiple trees, robust to noise", 
        "Reduces variance, improves stability and accuracy by averaging multiple models", 
        "Boosts the performance of weak learners, interpretable", 
        "Combines strengths of multiple models, can improve overall performance", 
        "Can capture complex patterns and relationships, highly flexible", 
        "Can handle non-linear relationships, highly flexible", 
        "Excellent for image and video data, can capture spatial hierarchies in data", 
        "Excellent for sequential data, can capture temporal dependencies", 
        "Efficient with large datasets, can handle sparse data", 
        "Simple to implement, effective with well-separated classes", 
        "Can handle non-linear relationships, simple to implement", 
        "Extremely fast training, simple to implement", 
        "Balances between LDA and QDA, can handle different types of distributions"
    ],
    "Weaknesses": [
        "Not suitable for complex relationships, assumes linearity", 
        "Assumes features are independent, not suitable for complex relationships", 
        "Assumes features are independent, not suitable for continuous data", 
        "Assumes features are independent, not suitable for continuous data", 
        "Slow with large datasets, sensitive to irrelevant features", 
        "Computationally intensive, sensitive to parameter settings", 
        "Prone to overfitting, unstable (small changes in data can lead to different trees)", 
        "Harder to interpret than individual decision trees, can be slow to train", 
        "Sensitive to overfitting, requires careful tuning of parameters", 
        "Sensitive to hyperparameters, less interpretable", 
        "Requires GPU for best performance, can be slower to train than LightGBM", 
        "Complex tuning process, high memory usage", 
        "Harder to interpret than individual decision trees, can be slow to train", 
        "Can be computationally intensive, less interpretable", 
        "Sensitive to noisy data and outliers, can overfit", 
        "Complex to implement and tune, can overfit if not properly validated", 
        "Requires large datasets and significant computational resources, difficult to interpret", 
        "Requires significant computational resources, difficult to interpret", 
        "Requires significant computational resources, complex to implement", 
        "Can suffer from vanishing/exploding gradient problems, requires significant computational resources", 
        "Sensitive to feature scaling, requires careful tuning of learning rate", 
        "Assumes normal distribution of features, not suitable for non-linear relationships", 
        "Requires a lot of data to estimate parameters, assumes normal distribution of features", 
        "Can overfit, sensitive to number of neurons and regularization", 
        "Requires careful tuning of regularization parameters, less interpretable"
    ],
    "Any additional information": [
        "Regularization techniques like L1, L2 can be used to avoid overfitting", "", "", "", 
        "Choice of distance metric (Euclidean, Manhattan, etc.) affects performance", 
        "Different kernel functions (linear, polynomial, RBF, etc.) can be used to enhance performance", 
        "Pruning techniques can be used to control overfitting", "", 
        "Regularization techniques can be used to avoid overfitting", 
        "Designed for performance and efficiency", "", "", "", "", "", "", 
        "Requires careful tuning of architecture and hyperparameters", "", "", 
        "Variants like LSTM and GRU can be used to mitigate gradient problems", "", "", "", "", ""
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Export the DataFrame to an Excel file
excel_path = 'Classifier_techniques.xlsx'
df.to_excel(excel_path, index=False)

print(f"The table has been exported to {excel_path}")

# # Export the DataFrame to an csv file
# csv_path = 'Classifier_techniques.csv'
# df.to_csv(excel_path, index=False)

# print(f"The table has been exported to {csv_path}")

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(24, 13))  # Set the figure size
ax.axis('tight')
ax.axis('off')

# Create the table plot with custom column widths
colWidths = [0.1, 0.045, 0.12, 0.12, 0.048, 0.065, 0.153, 0.17, 0.193, 0.177]  # Adjust the widths as needed
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
plt.savefig('classifier_techniques.png', bbox_inches='tight', dpi=300)

# Open the image using PIL to ensure it's saved correctly
img = Image.open('classifier_techniques.png')
img.show()





# data = {
#     "Technique": [
#         "Logistic Regression", "GaussianNB", "MultinomialNB", "BernoulliNB", 
#         "K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)", 
#         "Decision Tree (DT)", "Random Forest Classifier", 
#         "Gradient Boosting Classifier", "LightGBM", "CatBoost Classifier", 
#         "XGBoost Classifier", "Extra Trees Classifier", "Bagging Classifier", 
#         "AdaBoost Classifier", "StackingClassifier", "Neural Networks", 
#         "Multi-layer Perceptron (MLP) Classifier", 
#         "Convolutional Neural Networks (CNN) Classifier", 
#         "Recurrent Neural Networks (RNN) Classifier", 
#         "Stochastic Gradient Descent (SGD) Classifier", 
#         "Linear Discriminant Analysis (LDA) Classifier", 
#         "Quadratic Discriminant Analysis (QDA) Classifier", 
#         "Extreme Learning Machines (ELM) Classifier", 
#         "Regularized Discriminant Analysis (RDA) Classifier"
#     ],
#     "Coding Complexity": [
#         "Low", "Low", "Low", "Low", "Low", "Medium", "Low", "Medium", "Medium", 
#         "Medium", "Medium", "Medium", "Medium", "Medium", "Medium", "High", 
#         "High", "Medium", "High", "High", "Low", "Low", "Low", "Medium", "Medium"
#     ],
#     "Typical Use Cases": [
#         "Binary classification, Multi-class classification", 
#         "Text classification, Spam filtering", 
#         "Text classification, Document categorization", 
#         "Text classification, Spam detection", 
#         "Image classification, Recommendation systems", 
#         "Image classification, Text classification", 
#         "Customer segmentation, Medical diagnosis", 
#         "Fraud detection, Stock market analysis", 
#         "Web search ranking, Ecology", 
#         "Large-scale prediction tasks, Click-through rate prediction", 
#         "Recommendation systems, Forecasting", 
#         "Credit scoring, Ad click-through rate prediction", 
#         "Remote sensing, Bioinformatics", 
#         "Ensemble learning", 
#         "Face detection, Sentiment analysis", 
#         "Kaggle competitions, Complex prediction tasks", 
#         "Image and speech recognition, Natural language processing", 
#         "Pattern recognition, Speech recognition", 
#         "Image classification, Video analysis", 
#         "Time series prediction, Natural language processing", 
#         "Large-scale learning, Text classification", 
#         "Face recognition, Marketing", 
#         "Speech recognition, Medical diagnosis", 
#         "Bioinformatics, Image classification", 
#         "High-dimensional data classification"
#     ],
#     "When to Use": [
#         "Simple linear problems, interpretability needed",
#         "Small datasets, quick model building",
#         "Discrete features, text data",
#         "Binary/boolean features",
#         "Small to medium datasets, non-linear decision boundaries",
#         "High-dimensional data, clear margin of separation",
#         "Interpretable results needed, non-linear relationships",
#         "Large datasets, complex relationships",
#         "High predictive accuracy needed",
#         "Large datasets, fast training needed",
#         "Datasets with categorical features",
#         "When high performance is crucial",
#         "When further randomness is beneficial",
#         "To reduce variance of complex models",
#         "Boosting weak learners",
#         "When single models are not sufficient",
#         "Complex patterns, large datasets",
#         "Non-linear problems, sufficient data available",
#         "Image and grid-like data",
#         "Sequential data, time-dependent patterns",
#         "Very large datasets, online learning",
#         "Multiclass problems, dimensionality reduction needed",
#         "Non-linear decision boundary needed",
#         "Fast training needed, slightly lower accuracy acceptable",
#         "When LDA or QDA don't perform well"
#     ],
#     "Good with Overfitting": [
#         "No", "No", "No", "No", "Yes", "No", "Yes", "No", "No", "No", "No", 
#         "No", "No", "No", "No", "No", "Yes", "Yes", "Yes", "Yes", "No", "No", 
#         "No", "No", "No"
#     ],
#     "Good with Unbalanced Data": [
#         "No", "Yes", "Yes", "Yes", "No", "No", "Yes", "Yes", "No", "Yes", "Yes", 
#         "Yes", "Yes", "No", "No", "No", "No", "No", "No", "No", "No", "No", 
#         "No", "No", "No"
#     ],
#     "Strengths": [
#         "Simple, interpretable, fast training",
#         "Fast training and prediction, works well with high dimensions",
#         "Efficient for large datasets, works well with high dimensions",
#         "Good for short texts, works well with high dimensions",
#         "Simple, non-parametric, works with multi-class",
#         "Effective in high dimensional spaces, versatile with different kernels",
#         "Easy to understand, handles non-linear relationships",
#         "Handles large datasets, reduces overfitting",
#         "High performance, handles different types of data",
#         "Fast training speed, low memory usage, handles large datasets",
#         "Handles categorical features well, reduces overfitting",
#         "High performance, handles missing data",
#         "Faster training than Random Forest, good generalization",
#         "Reduces overfitting, improves stability",
#         "Can achieve high accuracy, less prone to overfitting",
#         "Can achieve higher accuracy than individual models",
#         "Can model complex non-linear relationships, versatile",
#         "Can approximate any continuous function",
#         "Excellent for spatial data, reduces parameters",
#         "Good for sequential and time-series data",
#         "Efficient for very large datasets, supports online learning",
#         "Works well for multi-class problems, can be used for dimensionality reduction",
#         "Doesn't assume common covariance matrix, works well with non-linear boundaries",
#         "Extremely fast training, good generalization",
#         "Balances between LDA and QDA, handles high-dimensional data well"
#     ],
#     "Weaknesses": [
#         "Assumes linearity, can't capture complex relationships",
#         "Assumes feature independence, sensitive to irrelevant features",
#         "Assumes feature independence, sensitive to feature scaling",
#         "Assumes feature independence, loses information in non-binary data",
#         "Slow for large datasets, sensitive to irrelevant features",
#         "Can be slow on large datasets, sensitive to feature scaling",
#         "Prone to overfitting, can be unstable",
#         "Less interpretable than single decision trees, computationally intensive",
#         "Can overfit if not tuned properly, computationally intensive",
#         "Can overfit on small datasets, requires careful parameter tuning",
#         "Can be slower than other boosting algorithms",
#         "Can be computationally expensive, requires careful tuning",
#         "Less accurate than Random Forest on some problems",
#         "Can be computationally expensive, may sacrifice interpretability",
#         "Sensitive to noisy data and outliers",
#         "Complex to implement, risk of overfitting",
#         "Requires large amounts of data, computationally intensive",
#         "Sensitive to feature scaling, requires hyperparameter tuning",
#         "Requires large datasets, computationally intensive",
#         "Can suffer from vanishing/exploding gradients",
#         "Requires feature scaling, sensitive to feature selection",
#         "Assumes normal distribution, sensitive to outliers",
#         "Requires more parameters than LDA, sensitive to small sample sizes",
#         "Less accurate than some other methods, sensitive to hidden layer size",
#         "Requires careful tuning of regularization parameters"
#     ],
#     "Additional Information": [
#         "Often used as a baseline model",
#         "Good for real-time prediction",
#         "Often used in natural language processing",
#         "Useful when feature presence/absence is important",
#         "Requires feature scaling, choice of K is crucial",
#         "Kernel choice is important",
#         "Good for feature importance analysis",
#         "Good for feature importance ranking",
#         "Requires careful parameter tuning",
#         "Good for categorical features",
#         "Good for datasets with categorical variables",
#         "Often wins machine learning competitions",
#         "Good for very high dimensional feature spaces",
#         "Often used with decision trees as base estimators",
#         "Good for combining weak classifiers",
#         "Requires careful selection of base models and meta-classifier",
#         "Many architectures available for different tasks",
#         "Good for learning complex patterns",
#         "Often used in computer vision tasks",
#         "Often used with LSTM or GRU units",
#         "Good when computational resources are limited",
#         "Good when you need probabilistic outcomes",
#         "Good when classes have different covariances",
#         "Good for real-time applications",
#         "Good for high-dimensional data with small sample sizes"
#     ]
# }


