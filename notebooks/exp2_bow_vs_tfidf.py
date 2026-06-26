import setuptools
import os
import re
import string
import pandas as pd
pd.set_option('future.no_silent_downcasting',True)
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse
import warnings
warnings.simplefilter("ignore",UserWarning)
warnings.filterwarnings("ignore")

# ---------------configuration--------------------
CONFIG ={
    "data_path": "notebooks/data.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "https://dagshub.com/raushankumar733329/MLOPS-CAPSTONE-PROJECT.mlflow",
    "dagshub_repo_owner": "raushankumar733329",
    "dagshub_repo_name":"MLOPS-CAPSTONE-PROJECT",
    "experiment_name":"Bow vs TfIdf"
}

#-------------------setup mlflow and dagshub
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"],repo_name=CONFIG["dagshub_repo_name"],mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])

#---------------text preprocessing----------------

def lemmatization(text):
    """Lemmatize the text"""
    lemmatizer =WordNetLemmatizer()
    text=text.split()
    text =[lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """remove stop words"""
    stop_words=set(stopwords.words("english"))
    text=[word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """remove numbers from the text"""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """convert text to lower case"""
    return text.lower()

def removing_punctuation(text):
    """remove punctuation from text"""
    return re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)


def removing_urls(text):
    """remove urls from the text"""
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalize_text(df):
    """normalize the text data"""
    try:
        df['review']=df['review'].apply(lower_case)
        df['review']=df['review'].apply(remove_stop_words)
        df['review']=df['review'].apply(removing_numbers)
        df['review']=df['review'].apply(removing_punctuation)
        df['review']=df['review'].apply(removing_urls)
        df['review']=df['review'].apply(lemmatization)
        return df
    except Exception as e:
        print(f'error during the text normalization: {e}')
        raise

#------------------Load and preprocess Data-----------------
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df = normalize_text(df)
        df = df[df['sentiment'].isin(['positive', 'negative'])]
        df['sentiment'] = df['sentiment'].replace({'negative': 0, 'positive': 1}).infer_objects(copy=False)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

#-------------------feature engineering---------------
VECTORIZERS = {
    'BoW': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

ALGORITHMS = {
    'LogisticRegression': LogisticRegression(),
    'MultinomialNB': MultinomialNB(),
    'XGBoost': XGBClassifier(),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}

#--------------train and evaluate models--------------
def train_and_evaluate(df):
    with mlflow.start_run(run_name="All Experiments") as parent_run:
        for algo_name,algorithm in ALGORITHMS.items():
            for vec_name,vectorizer in VECTORIZERS.items():
                with mlflow.start_run(run_name=f"{algo_name} with {vec_name}",nested=True) as child_run:
                    try:
                        # feature extraction
                        X=vectorizer.fit_transform(df['review'])
                        y=df['sentiment']
                        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=CONFIG["test_size"],random_state=42)

                        # log preprocessing parameters
                        mlflow.log_params({
                            "vectorizer": vec_name,
                            "algorithm": algo_name,
                            "test_size": CONFIG["test_size"]
                        })

                        # train model
                        model=algorithm
                        model.fit(X_train,y_train)

                        # log model parameters
                        log_model_params(algo_name, model)

                        # evaluate model
                        y_pred=model.predict(X_test)
                        metrics={
                            "accuracy":accuracy_score(y_test,y_pred),
                            "precision" :precision_score(y_test,y_pred),
                            "recall": recall_score(y_test,y_pred),
                            "f1_score":f1_score(y_test,y_pred)
                        }
                        mlflow.log_metrics(metrics)

                        # log model
                        #mlflow.sklearn.log_model(model,"model")
                        input_example=X_test[:5] if not scipy.sparse.issparse(X_test) else X_test[:5].toarray()
                        mlflow.sklearn.log_model(model, "model", input_example=input_example)

                        # print result for verification
                        print(f"\nAlgorithm: {algo_name}, Vectorizer: {vec_name}")
                        print(f"Metrics: {metrics}")

                    except Exception as e:
                        print(f"Error in training {algo_name} with {vec_name}:{e}")
                        mlflow.log_param("error",str(e))

def log_model_params(algo_name,model):
    """logs hyperparameters of the trained model to mlflow"""

    params_to_log={}
    if algo_name=="LogisticRegression":
        params_to_log["C"]=model.C

    elif algo_name=="MultinomialNB":
        params_to_log["alpha"]=model.alpha

    elif algo_name=='XGBoost':
        params_to_log['n_estimators']=model.n_estimators
        params_to_log["learning_rate"]=model.learning_rate

    elif algo_name=="RandomForest":
        params_to_log['n_estimators']=model.n_estimators
        params_to_log["max_depth"]=model.max_depth
        params_to_log["learning_rate"]=model.learning_rate

    mlflow.log_params(params_to_log)

# ----------------------Execution-------------------
if __name__=="__main__":
    df=load_data(CONFIG["data_path"])
    train_and_evaluate(df)