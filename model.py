import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

def train_xgboost():

    # Read data and setup input and output to the model
    df = pd.read_csv('data.csv')
    X = df.drop(columns=['timestamp', 'target'])
    y = df['target']

    # Split the data into train and test, has to be split by time as otherwise could cause a data leak (we will have some information about the future in the train set)
    split_index = int(len(df) * 0.85)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]


    # Setup and train the model
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=2,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)
    return model, X_test, y_test
