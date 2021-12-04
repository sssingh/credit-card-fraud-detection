# Credit Card Fraud Detection

![](images/main-project-image.jpg)

> ### Credit card fraud detection using machine learning techniques.

---

## Table of Contents

- [Introduction](#introduction) 
- [Objective](#objective)
- [Dataset](#dataset)
- [Solution Approach](#solution-approach)
- [How To Use](#how-to-use)
- [License](#license)
- [Author Info](#author-info)

---

## Introduction

Credit card fraud is a form of identity fraud/theft. It happens when unauthorized transaction is made using a credit card without
explicit permission from the card holder. There are various ways card fraud can be conducted...
- Fraudster can get discarded receipts or credit card statements that include your account number, and use that information to rack up fraudulent charges.
- Credit card info can be leaked during an online transaction and then unauthorized purchases can be conducted using this
information.
- Credit card information can be stolen using card skimmer installed at ATMs

Card fraud is a big problem for card issuers and banks as it account for substantial chunk of revenue loss. As per a report, fraud
losses reached up to USD 27.8 billion in 2019 and expect to go up to USD 35.67 billion in next 5 years. 

---
## Objective
In order to implement proactive monitoring, prevention mechanisms and reduce time consuming manual reviews and human errors this project aims to build and train a machine learning model to predict fraudulent credit card transaction.

---
## Dataset
- Dataset contains credit card transactions done by European card holders in 2013 over a period of 2 days
- There are 284,807 total transactions
- 492 transactions are fraudulent 
- The positive class (frauds) are only 0.172%
- Dataset has 31 features, 30 independent and 1 dependent column.
- Due to confidentiality reasons 28 features out of 30 independent features are transformed to numerical values using `PCA`. The remaining two features `time` and `amount` are left intact.  
- Fraudulent transactions are marked as 1 (Positive class) and genuine transactions are marked as 0 (negative class)
---

## Solution Approach
- We start with Exploratory Data Analysis to get in-depth data understanding, how predictors are affecting/not affecting the target variable and if given predictor are useful to us for model building.
- We then move to Data preparation where we try to remove skewness from data, split the data in test and train splits and apply standardization required.
- First we build various types of models like `LogisticRegression`, `DecisionTree`, `LinearSVC`, `RandomForest`, and `XGBoost` using the imbalanced dataset.
- Models are cross-validated using a 5 fold cross-validation to tune its hyper-parameters then re-fit the model using the best parameters obtained.
- From all the best models selected we select the best model using various metrics.
- `roc_auc` is used as primary metric for model evaluation and performance comparison. We also monitor and report
the `precision` and `recall` metrics for each of the models we build as they can give useful insights for different use-cases
- Depending upon the organization’s objective `precision` or `recall` could be one of the very important metrics for selecting a model. 
    - If organization is looking to catch most of the frauds while compromising on false-alarm where genuine
    transactions are classified as fraud then we’ll monitor ‘recall’. 
    - If reducing the false-alarm' is the primary concern then
    `precision` could be the go to metric.
- For this exercise we have taken the view that amount lost in fraudulent transactions is very high hence we’d like to catch as many of them as possible. Hence we use `recall` in addition to `roc-auc` as our primary performance metrics.
- We want our model prediction to be as interpretable as possible. This is one of the criteria's for selecting the final
model. Although we check the top most important features of the model, however due to PCA transformed features it’s hard to get any deeper insight on the important features.
- We then generate balanced dataset using `RANDOM`, `SMOTE` & `ADASYSN` oversampling techniques.
- All models built using imbalanced dataset are now fitted on to balanced dataset generated using above techniques.
- Best model of each type is then selected using the same metrics we use for imbalanced dataset.
- We then select the best model out of group of best models built on balanced dataset.
- Finally we compare the best model selected using imbalanced data vs the best model selected using balanced dataset and
arrive at the final-model for our project.

---

## How To Use
1. Ensure below listed packages are installed
    - `sklearn v1.0`
    - `numpy`
    - `pandas`
    - `xgboost`
2. Download `Credit_Card_Fraud_Detection.ipynb` jupyter notebook from this repo
3. Download dataset *creditcard.csv* from [here](https://drive.google.com/file/d/1aB4exBKkYppBkfJ-8eByy3pe4k6YC9bu/view?usp=sharing) in same folder where `Credit_Card_Fraud_Detection.ipynb` is kept
4. Open `Credit_Card_Fraud_Detection.ipynb` notebook and execute till section... [TODO]
5. Since some models will take longer to train its advisable to execute model training code one by one and save the trained model so that if training process halts in between then models already trained are not lost.

    > Save trained Model...[TODO]
    ```python
    import save_model

    model.save_model('file_name')

    ```

    > Load saved Model...[TODO]
    ```python
    import load_model

    model = load_model('file_name')

    ```
6. Predict using trained model [TODO]
    ```python
    preds = model.predict()
    ```
---

## Credits

- Title photo by [Ales Nesetril On Unsplash](https://unsplash.com/photos/ex_p4AaBxbs?utm_source=unsplash&utm_medium=referral&utm_content=creditShareLink)
- Data collected and analysed by [Worldline and the Machine Learning Group](http://mlg.ulb.ac.be) 
- Dataset sourced from [Kaggle](https://www.kaggle.com/)
---

## License

MIT License

Copyright (c) [2021] [Sunil S. Singh]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Author Info

- Twitter - [@sunilssingh6](https://twitter.com/sunilssingh6)
- Linkedin - [Sunil S. Singh](https://linkedin.com/in/sssingh)

[Back To The Top](#Credit-Card-Fraud-Detection)

---
