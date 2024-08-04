'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: pred_universe, num_fel_arrests_last_year
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.linear_model import LogisticRegression as lr
from sklearn.linear_model import LinearRegression


# Your code here
def logRegression(df):
    #Read in `df_arrests`- 

    # Separating the target variable
    #X = df_arrests #.values[:, 1:15]
    #Y = df_arrests #.values[:, 1:15]

    X=df[['num_fel_arrests_last_year','current_charge_felony']]
    Y=df[['y']]

        # Splitting the dataset into train and test
    X_df_arrests_train, X_df_arrests_test, y_df_arrests_train, y_df_arrests_test = train_test_split(
            X, Y, test_size=0.3, shuffle=True, stratify=Y)
        
    features = ['num_fel_arrests_last_year','current_charge_felony']

    #print(X.head())
    #print(Y.head())

    #print(X_df_arrests_train.head())
    #print(X_df_arrests_test.head())

    param_grid = {'C': [1, 10, 100], 'kernel': ['linear']}
                  
    #Initialize the Logistic Regression model with a variable called `lr_model` 
    #lr_model = LinearRegression().fit(X_df_arrests_train, y_df_arrests_train)
    lr_model = LinearRegression().fit(X_df_arrests_train, y_df_arrests_train)

    #Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
    #gs_cv = GridSearchCV(estimator=lr_model, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
    gs_cv = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=5)


    #Run the model 
    gs_cv.fit(X_df_arrests_test, y_df_arrests_test)

    print("Best Hyperparameters:", gs_cv.best_params_)
    print("Best Score:", gs_cv.best_score_)

    #return (X_df_arrests_test, y)