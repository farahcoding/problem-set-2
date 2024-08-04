'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import etl
import preprocessing
import logistic_regression
import decision_tree
import calibration_plot


# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    etl.getData()
    
    # PART 2: Call functions/instanciate objects from preprocessing
    df_arrests= preprocessing.getArrestDF()

    # PART 3: Call functions/instanciate objects from logistic_regression
    df_arrests_test,df_arrests_train, label_test,label_train, dfpred_lr =logistic_regression.logRegression(df_arrests)    

    # PART 4: Call functions/instanciate objects from decision_tree
    dfpred_dt=decision_tree.decisionTree(df_arrests_test,df_arrests_train, label_train)

    # PART 5: Call functions/instanciate objects from calibration_plot
    calibration_plot.calibration_plot(label_test, dfpred_lr.pred_lr ,5)
    calibration_plot.calibration_plot(label_test, dfpred_dt.pred_dt ,5)

if __name__ == "__main__":
    main()