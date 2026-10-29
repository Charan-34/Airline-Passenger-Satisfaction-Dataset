# Airline Passenger Satisfaction Analysis

## Project Overview

This project analyzes airline passenger satisfaction using Python. It explores passenger details, inflight services, online boarding, and flight delays to understand which factors are related to passenger satisfaction.

The project also uses machine learning models to predict passenger satisfaction and compare their performance.

## Objectives

1. Analyze overall passenger satisfaction.
2. Compare the average age of satisfied and dissatisfied passengers.
3. Analyze inflight Wi-Fi and entertainment ratings.
4. Study satisfaction rates based on online boarding ratings.
5. Compare departure and arrival delays by passenger satisfaction.

## Dataset

- **File name:** `data.csv`
- **Rows:** 129,880
- **Columns:** 22
- **Target column:** `satisfaction`

The dataset contains passenger information, travel details, service ratings, flight delays, and satisfaction labels.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Data Preprocessing

- Loaded the dataset using Pandas.
- Checked dataset information and summary statistics.
- Checked and removed duplicate records.
- Checked missing values and filled missing arrival-delay values using the median.
- Converted categorical features into numerical features using one-hot encoding.
- Split the data into training and testing sets.
- Applied feature scaling for machine learning.

## Exploratory Data Analysis (EDA)

The project includes graphs to understand:

- Overall passenger satisfaction.
- Passenger age and satisfaction.
- Inflight Wi-Fi and entertainment ratings.
- Satisfaction rate by online boarding rating.
- Departure and arrival delays.
- Correlations between numerical features.

## Machine Learning Models

Five classification models are used:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Decision Tree Classifier
5. Random Forest Classifier

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report

A model accuracy comparison graph is also used to compare the performance of all five models.

## Results

The performance of each model is compared to identify the model that performs best on the test dataset.

The final results depend on the dataset preprocessing, training/testing split, and model settings used when running the code.

## How to Run the Project

1. Clone this repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   ```

2. Open the project folder:

   ```bash
   cd YOUR_PROJECT_FOLDER
   ```

3. Install the required libraries:

   ```bash
   pip install pandas matplotlib seaborn scikit-learn
   ```

4. Make sure `data.csv` is in the project folder.

5. Run the Python file:

   ```bash
   python ca2project.py
   ```

## Conclusion

This project combines data analysis, visualization, and machine learning to study airline passenger satisfaction. It compares five classification models and evaluates their performance using multiple classification metrics.

## Author

Y.Charan
