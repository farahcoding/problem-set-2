'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Return dataframe(s) for use in main.py for PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 5 in main.py
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC


def decisionTree(df1, df2,df3):
    df_arrests_test = df1
    df_arrests_train = df2
    label_train = df3

    #Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
    param_grid_dt = {'max_depth': [1, 3, 5], 
                     'min_samples_split':range(1,10),
                     'min_samples_leaf':range(1,5),
                     'criterion': ['gini','entropy']}

    #Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
    dt_model = DTC()#.fit(df_arrests_train, df_arrests_train)

    #Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
    gs_cv_dt = GridSearchCV(estimator=dt_model, param_grid=param_grid_dt, cv=5)

    #Run the model 
    gs_cv_dt.fit(df_arrests_train, label_train)

    print("\nBest Hyperparameters:", gs_cv_dt.best_params_)
    #print("it was the least value.\n")
    #print("Best Score:", gs_cv_dt.best_score_)

    df_arrests_test['pred_dt'] = gs_cv_dt.best_estimator_.predict(df_arrests_test)

    #print(X_df_arrests_test.head(10))

    return (df_arrests_test)