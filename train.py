import os
import argparse
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset


web_path = "https://raw.githubusercontent.com/JoanneJons/azure-machine-learning-capstone/main/breast-cancer-dataset.csv?token=AJ5V2OGXYLJ22BGYXN4EUODAC6P4K"

def split_data(data):

    y_df = data['diagnosis']
    data.drop(['diagnosis'], inplace=True, axis=1)
    data.drop(['Unnamed: 0'], inplace=True, axis=1)
    x_df = data

    return x_df, y_df


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="Number of trees in the forest")
    parser.add_argument('--min_samples_split', type=int, default=2, help="Minimum number of samples required to split an internal node")
    parser.add_argument('--max_features', type=str, default='auto', help="{'auto', 'sqrt', 'log2'}")
    parser.add_argument('--bootstrap', type=bool, default=True, help="Whether bootstrap samples are used or not")


    args = parser.parse_args()

    ds = TabularDatasetFactory.from_delimited_files(path=web_path)

    x, y = split_data(ds)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    run = Run.get_context()

    run.log("No of Estimators:", np.int(args.n_estimators))
    run.log("Min No of Samples to Split:", np.int(args.min_samples_split))
    run.log("No of Features Considered:", np.str(args.max_features))
    run.log("Bootstrap:", np.bool(args.bootstrap))

    model = RandomForestClassifier(n_estimators=args.n_estimators, min_samples_split=args.min_samples_split, bootstrap=args.bootstrap, max_features=args.max_features).fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('output', exists_ok=True)
    joblib.dump(value=model, filename='outputs/model.pkl')

if __name__ == '__main__':
    main()


