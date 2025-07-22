# Multi-Output Deep Learning on Hotel Reviews 🏨🧠

## Aim of the project 🎯  
The aim of this project was to design a fully self-contained deep learning pipeline to predict both the sentiment (good/bad) and the numerical score of hotel reviews. The challenge required integrating structured features and free-text, and demonstrating best practices in model design, evaluation, and reproducibility as developed during the course.

## Description 📝  
This project uses a real-world dataset of hotel reviews containing both tabular data (e.g., number of reviews, reviewer nationality) and free-text fields (the review itself). The objective was to develop a hybrid deep neural network architecture capable of multi-output prediction:  
- **Binary classification** for the review sentiment (“good” or “bad” review)
- **Regression** for the review’s numerical score

The pipeline covers all steps from data cleaning and preprocessing (handling categorical, numerical, and textual features), through exploratory analysis, to building and tuning a multi-branch model (LSTM for text, MLP for structured features) with joint optimization for both outputs. Evaluation is rigorous and based on best practices in machine learning and deep learning.

### Objectives:
- Clean and preprocess data, standardizing numerical features, tokenizing and embedding review text, and encoding categorical variables  
- Drop potentially bias-inducing features (e.g., reviewer nationality), and motivate all design decisions  
- Design a multi-input, multi-output neural network: LSTM for text, MLP for tabular data, with merged features and separate output heads  
- Optimize and validate the model for both binary classification (review type) and regression (review score)  
- Visualize results via learning curves, confusion matrix, and report key metrics

### Main Results:  
- Achieved high accuracy (>90%) in predicting review sentiment and low mean squared error for score regression  
- Demonstrated robust generalization on unseen data, with stable learning curves and strong performance metrics on test set  
- Delivered a fully self-contained, well-documented Colab notebook, compliant with academic standards for reproducibility and explainability

## Technologies Used 💻  
- **Python**  
- **pandas** (data manipulation)  
- **NumPy** (numerical operations)  
- **TensorFlow / Keras** (LSTM, Dense/MLP, multi-output models, regularization, training)  
- **scikit-learn** (preprocessing, metrics, train/test split, cross-validation)  
- **matplotlib** & **seaborn** (visualization)  
- **Google Colab** (cloud notebook development, data pipeline, reproducibility)
