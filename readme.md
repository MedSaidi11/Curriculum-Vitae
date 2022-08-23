# **PROJECT REPORT (TEAM-A)**

**Saidi Mohamed (** **Team Lead)**

**Aruneshwar N (Co-Team** **Lead)**

**Muskan Dalal**

**Vinay Kumar Kushwaha**

**Rozmin Nisar Waghu**

**Suman Gautam**

**TOPIC**

**Predictive Modelling Using Social Profile in Online P2P Lending Market**

**INTRODUCTION**

Online peer-to-peer (P2P) lending markets enable individual consumers to borrow from and lend money to, one another directly. We study the borrower-loan- and group-related determinants of performance predictability in an online P2P lending market by conceptualizing financial and social strength to predict ROI and Loan status.

**DATASET USED:**

This data set contains 113,937 rows with 81 variables columns, including loan amount, borrower rate (or interest rate), current loan status, borrower income, and many others.

The original data can be found here: [https://s3.amazonaws.com/udacity-hosted-downloads/ud651/prosperLoanData.csv](https://s3.amazonaws.com/udacity-hosted-downloads/ud651/prosperLoanData.csv).

Data dictionary to understand the variables more from this link:

[https://docs.google.com/spreadsheets/d/1gDyi\_L4UvIrLTEC6Wri5nbaMmkGmLQBk-Yx3z0XDEtI/edit?usp=sharing](https://docs.google.com/spreadsheets/d/1gDyi_L4UvIrLTEC6Wri5nbaMmkGmLQBk-Yx3z0XDEtI/edit?usp=sharing).

**PYTHON LIBRARIES USED:**

NumPy, Pandas, Matplotlib, Seaborn, Sklearn, Flask

**MODEL USED:**

Logistic Regression, Decision Tree, Naive Bayes

**Data cleaning and imputing:**

The Dataframe has 81 columns of which 61 columns are numerical values while others are categorical variables. There are 43 columns that have missing values.

We will impute the missing values with KNN or mean for numerical values while mode for categorical variables with mode.

We will select some necessary variables which are important for finding Loan status by relating to the high correlation with them and drop other variables which are not important.

Loan Status has to Encode all completed loans as 1, and all delinquent, charged-off, cancelled and defaulted loans as 0.
```
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
imp.fit(df_n)
df_num_imputed = imp.transform(df_n)
```

**EDA (EXPLORATORY DATA ANALYSIS)**

We have visualized and analysed the variables in univariate and bivariate exploration. And found out that:

- Q1. What is the most number of borrower's Credit Grade?

ANS. We found Credit Grade C has the most borrows i.e., 5649.

- Q2. Since there is so much low Credit Grade such as C and D, does it lead to a higher amount of delinquency?

ANS. This shows that people who were having delays or problems with their loans were automatically labelled as delinquents and so their credit grade will be shown to be affected in terms of being a C or D instead of AA, A and B.

- Q3. What is the highest number of Borrower Rate?

ANS. The highest number of Borrower Rate is between 0.1 and 0.2 i.e., 25000 and above.

- Q4. Since the highest number of Borrower Rate is between 0.1 and 0.2, does the highest number of Lender Yield is between 0.1 and 0.2?

ANS. The highest number of Lender Yields is between 0.1 and 0.2 about 25000 and above.

- Q5. Is the Credit Grade really accurate? Does a higher Credit Grade lead to higher Monthly Loan Payment?

ANS. The Credit Grades are really accurate. Higher Credit Grade A, AA and followed by B are leading to pay higher monthly payments.

- Q6. Here we look at the Completed Loan Status and Defaulted Rate to determine the accuracy of Credit Grade.

ANS. Total Loan completed by Credit Grade C are 3609 out of 5649 which is the most.

Total Loan Default by Credit Grade HR is 891 out of 1372 which is the most. Followed by Credit grade C has 729 out of 5649. While least default by higher Credit grades AA and A have 201 out of 3509 and 283 out of 3315 respectively.

- Q7. Now we know the Credit Grade is accurate and is a tool that is used by the organization in determining a person's creditworthiness. Now we need to understand the Prosper Score, the custom-built risk assessment system being used in determining borrower rates.

ANS. Prosper Score is worth for credit borrowing rate. As Prosper Score is 1.0 for high borrows rate and BorrowsAPR while Prosper Score is 11.0 for low borrows rate and BorrowsAPR.

**FEATURE ENGINEERING:**

Selected some variables for feature engineering using lightgbm for feature importance. Features are separated into two parts for categorical and numerical variables types.

```
gbm.booster_.feature_importance()
fea_imp_ = pd.DataFrame({'cols':X1.columns, 'fea_imp':gbm.feature_importances_})
fea_imp_sorted = fea_imp_.loc[fea_imp_.fea_imp > 0].sort_values(by=['fea_imp'], ascending = True)
fea_imp_sorted
```
**#Data Pre-processing through Onehotencoder and column transformer for numerical features:**

The next process outlined demonstrates to one-hot encode a single column. Sklearn comes with a helper function, make\_column\_transformer() which aids in the transformations of columns. The function generates ColumnTransformer objects for you and handles the transformations for the numerical features.

**# Data Pre-processing through Labelencoder for categorical features:**

Label Encoding in Python can be achieved using Sklearn Library. Sklearn provides a very efficient tool for encoding the levels of categorical features into numeric values. LabelEncoder encode labels with a value between 0 and n\_classes-1 where n is the number of distinct labels. If a label repeats it assigns the same value to as assigned earlier.
```
from sklearn.preprocessing import LabelEncoder
cat_list = [] 
num_list = []
for colname, colvalue in df_c.iteritems():
        cat_list.append(colname)
for col in cat_list:
    encoder = LabelEncoder()
    encoder.fit(df_c[col])
    df_c[col] = encoder.transform(df_c[col])
```

**#StandardScalar:**

StandardScaler is that it will transform your data such that its distribution will have a mean value 0 and standard deviation of 1. In the case of multivariate data, here is done feature-wise (in other words independently for each column of the data). Given the distribution of the data, each value in the dataset will have the mean value subtracted and then divided by the standard deviation of the features.

**#Hyper parameter tuning:**

Light GBM is a fast, distributed, high-performance gradient boosting framework based on decision tree algorithm, used for ranking, classification and many other machine learning tasksLight GBM uses leaf wise splitting over depth-wise splitting which enables it to converge much faster but also leads to overfitting. Here we will apply LGBM regressor on the X and y dataset.

**#Spliting the training and testing dataset:**

We will divide the X as independent variables from features and y as dependent variable Loan status into four parts i.e., X\_train, y\_train , X\_test and y\_test.

**Logistic regression model :**

- Logistic Regression helps find how probabilities are changed with actions.

- The function is defined as P(y) = 1 / 1+e^-(A+Bx)

- Logistic regression involves finding the \*\*best fit S-curve\*\* where A is the intercept and B is the regression coefficient. The output of logistic regression is a probability score.

**#Metrics considered for Model Evaluation:**

Accuracy: What proportion of actual positives and negatives is correctly classified?

Root Mean Square Error (RMSE) : RMSE is the square root of the value obtained from the Mean Square Error between the estimated and actual values of a parameter of the model.
```
from sklearn.linear\_model import LogisticRegression
from sklearn.metrics import mean\_squared\_error,accuracy\_score
df\_models = pd.DataFrame(columns=['model', 'run\_time', 'rmse', 'rmse\_cv','acc'])
print('\*',"LogisticRegression")
start\_time = time.time()
regressor = LogisticRegression()
model = regressor.fit(X\_train, y\_train)
y\_pred = model.predict(X\_valid)
scores = cross\_val\_score(model,
X\_train,
y\_train,
scoring="neg\_mean\_squared\_error",
cv=5)
row = {'model': "LogisticRegression",
'run\_time': format(round((time.time() - start\_time)/60,2)),
'rmse': np.sqrt(mean\_squared\_error(y\_valid, y\_pred)),
'rmse\_cv': np.mean(np.sqrt(-scores)),
'acc': np.mean(accuracy\_score(y\_valid,y\_pred))
}
df\_models = df\_models.append(row, ignore\_index=True)
df\_models.head().sort\_values(by='rmse\_cv', ascending=True)
```

**Result based on Logistic regression:**

| Model | run\_time | rmse | rmse\_cv | acc |
| --- | --- | --- | --- | --- |
| LogisticRegression | 0.19 | 0.079862 | 0.076726 | 0.993622 |

**Decision tree classifier model :**

- In this kind of decision trees, the decision variable is categorical. The above decision tree is an example of classification decision tree.
- Such a tree is built through a process known as binary recursive partitioning. This is an iterative process of splitting the data into partitions, and then splitting it up further on each of the branches.
- Decisions are based on some conditions. Decision made can be easily explained.
- Decision trees can handle high dimensional data with good accuracy.

The tree contains decision nodes and leaf nodes.

- Decision nodes are those nodes represent the value of the input variable(x). It has two or more than two branches.
- Leaf nodes contain the decision or the output variable(y).
```
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean\_squared\_error,accuracy\_score
regressors = { "DecisionTreeClassifier": DecisionTreeClassifier(criterion='entropy') }
df\_models = pd.DataFrame(columns=['model', 'run\_time', 'rmse', 'rmse\_cv'])
for key in regressors:
print('\*',key)
start\_time = time.time()
regressor = regressors[key]
model = regressor.fit(X\_train, y\_train)
y\_pred = model.predict(X\_test)
scores = cross\_val\_score(model,
X\_train,
y\_train,
scoring="neg\_mean\_squared\_error",
cv=5)
row = {'model': key,
'run\_time': format(round((time.time() - start\_time)/60,2)),
'rmse': np.sqrt(mean\_squared\_error(y\_test, y\_pred)),
'rmse\_cv': np.mean(np.sqrt(-scores)),
'accuracy': np.mean(accuracy\_score(y\_test,y\_pred))
}
df\_models = df\_models.append(row, ignore\_index=True)
df\_models.head().sort\_values(by='rmse\_cv', ascending=True)
```

**Result based on Decision tree classifier:**

| Model | run\_time | rmse | rmse\_cv | accuracy |
| --- | --- | --- | --- | --- |
| DecisionTreeClassifier | 0.14 | 0.071349 | 0.074635 | 0.994909 |
| Logistic Regression    | 0.50 | 0.061808 | 0.084635 | 0.996354 |
| Gaussian NB | 0.70 | 0.077774 | 0.077974 | 0.994081 |


**Naïve Bayes Classifier model:**

Naïve Bayes Classifier uses Bayes' theorem to predict membership probabilities for each class such as the probability that a given record or data point belongs to a particular class. The class with the highest probability is considered the most likely class.

The formula for Bayes' theorem is given as:

P(A|B) = P(B|A) \* P(A) / P(B)

where,

- P(A|B) is Posterior probability: Probability of hypothesis A on the observed event B.
- P(B|A) is Likelihood probability: Probability of the evidence given that the probability of a hypothesis is true.
- P(A) is Prior Probability: Probability of hypothesis before observing the evidence.
- P(B) is Marginal Probability: Probability of Evidence.

There are 3 types of Naïve Bayes algorithm. The 3 types are listed below: -

- Gaussian Naïve Bayes
- Multinomial Naïve Bayes
- Bernoulli Naïve Bayes

Here we will use Gaussian naïve Bayes model for modeling. When we have continuous attribute values, we made an assumption that the values associated with each class are distributed according to Gaussian or Normal distribution.

We dropped some features and use the (feat) variables given below to increase accuracy for the Naïve Bayes model which has higher correlations with the Loan status variable. Again we will do data preprocessing and columns transformer using Labelcoder and Onehotcoder.

```
feat = data[['Term', 'LoanOriginalAmount','BorrowerAPR', 'BorrowerRate', 'LenderYield', 'LoanStatus' , 'BankcardUtilization' , 'IncomeRange', 'StatedMonthlyIncome', 'EmploymentStatus', 'BorrowerState', 'DebtToIncomeRatio' , 'ProsperRating (Alpha)', 'ProsperScore', 'CreditGrade']]
from sklearn.model\_selection import train\_test\_split
X\_train, X\_test, y\_train, y\_test = train\_test\_split(x, y, test\_size = 0.3, random\_state = 42)
from sklearn.naive\_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X\_train, y\_train)
y\_pred = gnb.predict(X\_test)
from sklearn.metrics import classification\_report
print(classification\_report(y\_test, y\_pred))
from sklearn.metrics import accuracy\_score
print('Model accuracy score: {:.2f}%' .format(accuracy\_score(y\_test, y\_pred)\*100))
from sklearn.metrics import confusion\_matrix
cm = confusion\_matrix(y\_test, y\_pred)
```
**Result based on Naïve bayes model :**

#Model evaluation

- Classification Report: Classification report is another way to evaluate the classification model performance. It displays the precision, recall, f1 and support scores for the model.
- Accuracy: What proportion of actual positives and negatives are correctly classified?
- Confusion matrix: It gives us a summary of correct and incorrect predictions broken down by each category.

| Classification Report | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| 0 | 0.39 | 0.45 | 0.42 | 5735 |
| 1 | 0.89 | 0.86 | 0.87 | 28447 |

From accuracy score metrics we found that, the Naïve Bayes Model accuracy score: 79.01%

From Confusion matrix metrics, we found that,

[[2599 3136]

[4039 24408]]

| True Positives(TP) = 2599 | False Positives(FP) = 3136 |
| --- | --- |
| True Negatives(TN) = 24408 | False Negatives(FN) = 4039 |

**MODEL DEPLOYMENT:**

**# Streamlit**

- It is a tool that lets you create applications for your machine learning model by using simple python code.

- We write a python code for our app using Streamlit; the app asks the user to enter the following data (\*\*LoanStatus\*\*, \*\*Defaulted\*\*, \*\*Completed\*\*).

- The output of our app will be 0 or 1 ; 0 indicates that Loan will default while 1 means Loan will be completed.

- The app runs on a local host.

- To deploy it on the internet we have to deploy it to Heroku.

**#Flask**

We will import flask in PyCharm and then create our app with help of using Flask in index.html and predict.html files.

**#Heroku**

We deploy our Streamlit app to [Heroku.com](https://www.heroku.com/). In this way, we can share our app on the internet with others.

We prepared the needed files to deploy our app successfully:

- Procfile: contains run statements for app file and setup.sh.

- setup.sh: contains setup information.

- requirements.txt: contains the libraries that must be downloaded by Heroku to run app file (model.py) successfully

-app.py: contains the python code of a Streamlit web app.

- model.pkl : contains all models that were built that used for prediction.

- model2.pkl: contains the train data of the modeling part that will be used to rate of investment.

Finally, you can access our app by following this [https://demoloanapp.herokuapp.com/](https://demoloanapp.herokuapp.com/)

