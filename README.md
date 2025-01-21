# Food Price Prediction Model

Food prices fluctuate due to various factors, such as changes in supply, demand, weather conditions, and political factors. Predicting food prices is crucial for businesses and consumers to make informed decisions. In this project, I built a machine learning model to predict food prices across various states in Nigeria, such as Lagos, Kano, Abuja, and Rivers, based on factors like the state, zone, food type, year, and month. The goal is to develop an accurate predictive model that can be deployed on the web for easy access and use.



---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Dataset Description](#dataset-description)
4. [Model Development Process](#model-development-process)
    - [Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
    - [Data Preprocessing](#2-data-preprocessing)
    - [Feature Engineering](#3-feature-engineering)
    - [Model Training](#4-model-training)
    - [Evaluation](#5-evaluation)
    - [Model Deployment](#6-model-deployment)
5. [Technologies Used](#technologies-used)
6. [Setup Instructions](#setup-instructions)
7. [Usage](#usage)
8. [Future Improvements](#future-improvements)

---

## Overview

Food prices fluctuate due to various factors such as supply, demand, weather, and political conditions. This project aims to create a predictive model using machine learning to estimate food prices based on input features. The model is deployed using **Streamlit**, allowing users to interactively predict food prices for specific food items in different regions and time periods.

---

## Features

- Predict food prices based on state, zone, food type, year, and month.
- Interactive web app built with **Streamlit**.
- Handles categorical features using one-hot encoding.
- Supports predictions for multiple food items and regions.
- Displays encoded user input and predicted price.

---

## Dataset Description

The dataset includes the following columns:

- **State**: State where the food item is sold (e.g., Lagos, Kano, Abuja).
- **Zone**: Geopolitical zone (e.g., S.East, N.West).
- **Food Type**: Name of the food item (e.g., "Rice imported high quality sold loose").
- **Year**: Year of interest.
- **Month**: Month of interest.
- **Price**: The target variable representing the price of the food item.

---

## Model Development Process

### 1. Exploratory Data Analysis (EDA)
- Analyzed the distribution of prices and other features.
- Checked for missing values and outliers.
- Visualized trends across states, zones, and time periods.

### 2. Data Preprocessing
- Handled missing values by filling or removing them.
- Encoded categorical variables using one-hot encoding.
- Scaled numerical features where necessary.

### 3. Feature Engineering
- Created meaningful features such as "season" if applicable.
- Encoded categorical features using pandas' `pd.get_dummies` method.
- Ensured all features matched the training data's expected input format.

### 4. Model Training
- Tested multiple regression algorithms:
  - Linear Regression
  - Random Forest Regression
  - Gradient Boosting
  - XGBoost
- Selected **Random Forest Regression** as the final model due to its high R-squared (R²) and low error metrics.

### 5. Evaluation
- Used metrics such as **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)**, and **R²** to evaluate model performance.
- Achieved the following performance for Random Forest:
  - MSE: 1415.35
  - RMSE: 37.62
  - MAE: 23.19
  - R²: 0.99

### 6. Model Deployment
- Deployed the trained Random Forest model using **Streamlit**.
- Built an interactive web app for users to input values and get predictions in real-time.

---

## Technologies Used

### Programming Languages
- Python

### Tools
- Jupyter Notebook
- VS Code
- Streamlit

### Frameworks and Libraries
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- joblib

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip package manager

### Installation Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/food-price-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd food-price-prediction
    ```
3. Install required libraries:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

---

## Usage

1. Open the Streamlit app in your browser.
2. Select the state, zone, food item, year, and month from the dropdown menus and sliders.
3. Click the **Predict Price** button to get the predicted food price.
4. View the encoded input data and the predicted price displayed on the app.

---

## Future Improvements

- Incorporate additional features such as weather conditions and market trends.
- Extend the dataset to include more states and food types.
- Optimize the model using hyperparameter tuning.
- Add interactive visualizations to display price trends and patterns.
- Deploy the app using cloud services like AWS, Azure, or Google Cloud.

---

Feel free to contribute to this project by submitting pull requests or opening issues for feature requests and bug fixes.

