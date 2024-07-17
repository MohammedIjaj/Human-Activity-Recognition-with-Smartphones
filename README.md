# Human-Activity-Recognition-with-Smartphones

This project involves performing Exploratory Data Analysis (EDA), feature engineering, model training, evaluation, and making predictions using provided datasets.


## Introduction

This project is aimed at analyzing and predicting human activities using sensor data from wearable devices. The analysis involves various steps including data loading, exploratory data analysis, feature engineering, model training, evaluation, and prediction.

## Installation

Ensure you have the following Python libraries installed:
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `matplotlib` and `seaborn`: For data visualization.
- `scikit-learn`: For machine learning models and evaluation.

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

### Loading the Data

Load the training and test datasets into pandas DataFrames. This is the first step to read the datasets and make them ready for analysis.

### Exploratory Data Analysis (EDA)

Perform exploratory data analysis to understand the dataset. This includes:
- **Checking for Missing Values**: Identify any missing values in the datasets to determine if any data cleaning is required.
- **Descriptive Statistics**: Calculate and display basic statistics such as mean, standard deviation, and distribution for key features.
- **Data Visualization**: Visualize the distribution of key features and the target variable (`Activity`) to get a better understanding of the data.

### Feature Engineering

Feature engineering involves selecting and transforming features to improve the performance of machine learning models. This can include:
- **Feature Selection**: Identify the most relevant features for the prediction task using techniques like ANOVA F-value.
- **Feature Transformation**: Apply necessary transformations to the features to make them suitable for model training.

### Model Training

Train machine learning models on the training data. This involves:
- **Splitting the Data**: Split the data into training and validation sets.
- **Training the Model**: Train a classification model (e.g., Random Forest) using the training data.
- **Hyperparameter Tuning**: Optimize the model's hyperparameters to achieve better performance.

### Model Evaluation

Evaluate the trained model's performance using appropriate metrics such as:
- **Classification Report**: Provides precision, recall, F1-score, and support for each class.
- **Confusion Matrix**: Displays the confusion matrix to understand the model's predictions better.

### Prediction on Test Data

Use the trained model to make predictions on the test dataset. Analyze the predicted activities and compare them with actual activities to assess the model's performance on unseen data.

## Results

Summarize the results of the analysis, including the performance of the trained models, insights gained from EDA, and the accuracy of predictions on the test dataset.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.
