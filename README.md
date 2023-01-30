# Credit Card Fraud Detection
In this project, we tackle a classic `binary-classification` problem. Our ultimate goal is to build and train machine-learning models to detect fraudulent credit card transactions.

<img src="images/main-project-image.jpg" width="600" height="300" />

## Features
⚡Binary Classification  
⚡Dataset Balancing (RANDOM/SMOTE/ADASYN)  
⚡LogisticRegression  
⚡DecisionTree  
⚡LinearSVC  
⚡RandomForest  
⚡XGBoost  
⚡Python  
⚡Scikit-Learn  

## Table of Contents

- [Introduction](#introduction) 
- [Objective](#objective)
- [Dataset](#dataset)
- [Solution Approach](#solution-approach)
- [Evaluation Criteria](#evaluation-criteria)
- [How To Use](#how-to-use)
- [License](#license)
- [Author Info](#get-in-touch)
- [Credits](#credits)


## Introduction

Credit card fraud is a form of identity fraud/theft. It happens when an unauthorized transaction is made using a credit card without
explicit permission from the cardholder. There are various ways card fraud can be conducted...
- Fraudsters can get discarded receipts or credit card statements that include your account number and use that information to rack up fraudulent charges.
- Credit card info can be leaked during an online transaction, and then unauthorized purchases can be conducted using this
information.
- Credit card information can be stolen using a card skimmer installed at ATMs

Card fraud is a big problem for card issuers and banks as it accounts for a substantial chunk of revenue loss. As per <a href="https://www.prnewswire.com/news-releases/payment-card-fraud-losses-reach-27-85-billion-300963232.html"> Annual Fraud Statistics Released by The Nilson Report</a>, fraud
losses reached USD 27.8 billion in 2019 and expect to go up to USD 35.67 billion in the next five years. 

## Objective
To implement proactive monitoring, prevention mechanisms, reduce time-consuming manual reviews and human errors, this project aims to build and train a machine learning model to predict fraudulent credit card transactions.


## Dataset
- Dataset contains credit card transactions done by European cardholders in 2013 for two days
- There are 284,807 total transactions
- 492 transactions are fraudulent 
- The positive class (frauds) are only 0.172%
- Dataset has 31 features, 30 independent and one dependent column.
- Due to confidentiality reasons, 28 out of 30 independent features are transformed to numerical values using `PCA`. The remaining two features, `time` and `amount,` are left intact.  
- Fraudulent transactions are marked as 1 (Positive class), and genuine transactions are marked as 0 (negative class)


## Solution Approach
- We start with Exploratory Data Analysis to get in-depth data understanding, how predictors are affecting/not affecting the target variable and if given predictors are helpful to us for model building.
We then move to Data preparation where we try to remove skewness from data, split the data in test, train splits, and apply the standardization required.
- First, we build various models like `LogisticRegression,` `DecisionTree,` `LinearSVC,` `RandomForest`, and `XGBoost` using the imbalanced dataset.
- Models are cross-validated using five-fold cross-validation to tune its hyper-parameters, then re-fit the model using the best parameters obtained.
We select the best model using various metrics from all the best models selected.
- `roc_auc` is the primary metric for model evaluation and performance comparison. We also monitor and report
the `precision` and `recall` metrics for each of the models we build as they can give valuable insights for different use-cases
- Depending upon the organization's objective, `precision` or `recall` could be one of the vital metrics for selecting a model. 
    - If the organization is looking to catch most of the frauds while compromising on false-alarm where genuine
    transactions are classified as fraud; then we'll monitor 'recall.' 
    - If reducing the false alarm is the primary concern, then
    `precision` could be the go-to metric.
- For this exercise, we have taken the view that the amount lost in fraudulent transactions is very high; hence we'd like to catch as many of them as possible. Therefore we use `recall` in addition to `roc-auc` as our primary performance metrics.
- We want our model prediction to be as interpretable as possible. The easy inference is one of the criteria for selecting the final
model. Although we check the top most important features of the model, it's hard to get any deeper insight into the essential features due to PCA transformed features.
- We then generate a balanced dataset using `RANDOM,` `SMOTE` & `ADASYSN` oversampling techniques.
All models built using an imbalanced dataset are now fitted to the balanced dataset generated using the above techniques.
The best model of each type is then selected using the same metrics we use for the imbalanced dataset.
- We then select the best model out of the best models built on the balanced dataset.
- Finally, we compare the best model selected using imbalanced data vs. the best model selected using a balanced dataset and
arrive at the final model for our project.

## Evaluation Criteria
* `roc_auc` will be used as the primary metric for model evaluation and performance comparison
* We'll also monitor the `precision` and `recall` for each of the models we build as they can give valuable insights for different use-cases

## How To Use
1. Ensure the below-listed packages are installed
    - `sklearn v1.0`
    - `NumPy`
    - `pandas`
    - `xgboost`
    - `imblearn`
2. Download `Credit_Card_Fraud_Detection.ipynb` jupyter notebook from this repo
3. Download the dataset *creditcard.csv* from [here](https://drive.google.com/file/d/1n_ddBvn2dThcYE2hnrXg3kM153dGQGxo/view?usp=sharing) in same folder where `Credit_Card_Fraud_Detection.ipynb` is kept
4. Notebook `Credit_Card_Fraud_Detection.ipynb` can be executed from start to finish in one go. However, given the amount of data and Hyperparameter tuning done using Gridsearch, the complete notebook takes around **~8 hrs** to run end to end on intel i7 4-cores 32Gb machine with n_job set to -1 (using all four cores while training). Hence, it is advisable to execute model training code one by one and save the trained model so that if the training process halts in between, models already trained are not lost.

To save the trained model, you can use the below code...
```python
    import pickle
    
    pickle.dump(model, open('model_name.pkl', 'wb'))

```

To load the saved model you can use below code...
```python
    model =pickle.load(open('model_name.pkl', 'rb'))

```
6. Predict using trained/loaded model
    ```python
    preds = model.predict()
    ```

## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

## Get in touch

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/sssingh)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/_sssingh)
[![website](https://img.shields.io/badge/website-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://datamatrix-ml.com/)


## Credits

- Title photo by [Ales Nesetril On Unsplash](https://unsplash.com/photos/ex_p4AaBxbs?utm_source=unsplash&utm_medium=referral&utm_content=creditShareLink)
- Data collected and analyzed by [Worldline and the Machine Learning Group](http://mlg.ulb.ac.be) 
- Dataset sourced from [Kaggle](https://www.kaggle.com/)

[Back To The Top](#Credit-Card-Fraud-Detection)
