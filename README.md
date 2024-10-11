# House Price Prediction

This project aims to build a model for predicting house prices in King County, USA, using linear regression. The dataset, sourced from Kaggle, includes multiple attributes of houses, such as their selling price.

### Objectives:
- **Data Exploration and Analysis**:  
   Perform statistical analysis, identify missing data, visualize feature relationships, and explore the correlation between various features and house prices.
  
- **Data Preprocessing**:  
   Prepare the dataset for modeling by managing categorical features, scaling numeric data, and dividing the data into training and testing sets.

- **Model Development**:  
   Create a linear regression model using the scikit-learn library. Train the model on the training dataset and predict house prices on the test dataset.

- **Model Evaluation**:  
   Evaluate model performance using metrics such as R-squared and Mean Squared Error (MSE).

- **OLS Regression Analysis**:  
   Conduct an Ordinary Least Squares (OLS) regression using the statsmodels library to analyze model coefficients and assess their statistical significance.

- **Model Enhancement**:  
   Review model residuals and propose improvements to refine prediction accuracy.

---

### Exploratory Data Analysis (EDA)

![Graph](https://github.com/NikitaKundle01/House-Price-prediction/blob/main/output.png?raw=true)

This scatter plot shows the relationship between house price and square footage of living space. The x-axis represents square footage (`sqft_living`), and the y-axis represents the price.

#### Key Observations:
- **Positive Correlation**: There's a clear positive relationship between price and square footage—larger houses tend to have higher prices.
- **Outliers**: A few data points deviate from the overall trend, indicating outliers.
- **Clustering**: Data points are clustered in specific areas, suggesting the presence of distinct market segments or price brackets.
- **Non-Linearity**: The relationship isn’t perfectly linear, implying that price increases may not be proportional to increases in square footage.

#### Additional Analysis:
- **Correlation Coefficient**: Calculating this value would quantify the strength of the price-square footage relationship.
- **Regression Analysis**: Fitting a regression line could yield a more precise equation to predict house prices based on square footage.

---

![Graph](https://github.com/NikitaKundle01/House-Price-prediction/blob/main/output1.png?raw=true)

This histogram illustrates the distribution of house prices. The x-axis shows price (in millions), and the y-axis shows the number of houses in each price range.

#### Key Observations:
- **Right-Skewed Distribution**: The distribution is skewed to the right, indicating that most houses have lower prices, with fewer high-priced homes.
- **Peak Around 1 Million**: The distribution peaks around 1 million, showing this is the most common price range.
- **Long Tail**: The long tail suggests the presence of high-priced houses, exceeding 5 million.
- **Skewness**: A skewness coefficient could quantify the right-skewed nature of the data.

---

![Graph](https://github.com/NikitaKundle01/House-Price-prediction/blob/main/output2.png?raw=true)

#### Key Observations:
- **Strong Positive Correlation**: `sqft_living` has a strong positive correlation with price, suggesting larger homes command higher prices.
- **Other Positive Correlations**: Attributes such as the number of bedrooms, bathrooms, and house grade also show positive correlations with price.
- **Negative Correlations**: Some features, like `zipcode` and `lat`, have negative correlations with price.
- **Weak Correlations**: Several variables have weak or negligible correlations, indicating they have little influence on house pricing.

#### Additional Analysis:
- **Correlation Coefficients**: These values provide a quantitative measure of correlation strength.
- **Hierarchical Clustering**: Applying this technique to the correlation matrix could reveal groups of strongly correlated variables.
- **Multiple Regression**: Using significant features in a multiple regression analysis would allow for the combined influence of multiple factors on house prices to be evaluated.

---

# House Price Prediction with Linear Regression

This project delves into predicting house prices in King County, USA, using linear regression. The "kc_house_data.csv" dataset, available on Kaggle, serves as the basis for building and evaluating the model.

## Project Overview

The objective is to predict house prices using various factors like square footage, number of bedrooms, and bathrooms. Key processes include data exploration, visualization, and the implementation of a machine learning model.

### Key Steps

1. **Data Loading and Exploration**:
    - Load the dataset into a pandas DataFrame.
    - Examine data types, missing values, and descriptive statistics.
    - Conduct EDA to uncover feature relationships, focusing on the correlation between `price` and `sqft_living`.
    - Create visualizations like scatter plots, histograms, and correlation matrices.
    - Prepare data by handling missing values, adjusting data types, and removing irrelevant columns.

2. **Machine Learning and Model Development**:
    - Split the dataset into training and testing sets.
    - Implement a linear regression model using scikit-learn’s `LinearRegression`.
    - Train the model and evaluate performance using metrics like Mean Squared Error (MSE) and R-squared.

3. **OLS Model (Optional)**:
    - Fit an OLS regression model using the `statsmodels` library.
    - Review a statistical summary, including coefficients, p-values, and confidence intervals.

---

## Prerequisites

Before running the notebook, ensure you have the following:

- **Python 3.x**
- Required libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - (Optional) `statsmodels` for OLS regression

---

## Installation

1. **Install Libraries**:  
   Run the following command to install necessary libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
