import warnings
import datetime
from datetime import datetime

import os
import sys

import pandas as pd
import numpy as np
import collections
import sys
import psycopg2
from psycopg2.extras import execute_values

import mlflow
from sklearn.metrics import accuracy_score, recall_score, f1_score, average_precision_score
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor as task

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def read_data():
    index_file = 0
    files = ['/data/jupyter/healthcare-dataset-stroke-data.csv']
    ds = pd.read_csv(files[index_file], index_col=0)
    return ds
    
def connect_postgress():
    """ open conexion to the PostgreSQL database"""
    conn = psycopg2.connect(
        host="postgres-mlflow",
        port=5432,
        database="mlflow_db",
        user="mlflow_user",
        password="mlflow")
    return conn
    
def prepare_values(ds,start,length):
    df = ds.reset_index()
    value = df.iloc[start:length].values
    print(value)
    values = []
    try:
        for i in range(length-start):
            print(np.array(value[i][0]))
            values.append(np.array(value[i]))
        values = list(values)
    except:
        values = "Full"
    return values
    
def create_tables():
    """ create tables in the PostgreSQL database"""
    commands = (
        """
        CREATE TABLE stroke (
            id BIGINT PRIMARY KEY,
            gender VARCHAR(20),
            age INT,
            hypertension INT,
            heart_disease INT,
            ever_married VARCHAR(3),
            work_type VARCHAR(200),
            Residence_type VARCHAR(20),
            avg_glucose_level FLOAT,
            bmi FLOAT,
            smoking_status VARCHAR(20),
            stroke INT
        )
        """,
        """
        CREATE TABLE results (
            id BIGINT,
            timestamp VARCHAR(100),
            stroke_pred INT
        )
        """)
    conn = None
    a = False
    try:
        # read the connection parameters
        #params = config()
        # connect to the PostgreSQL server
        #conn = psycopg2.connect(**params)
        conn = connect_postgress()
        cur = conn.cursor()
        # create table one by one
        for command in commands:
            cur.execute(command)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        a = True
    finally:
        if conn is not None:
            conn.close()
    return a
            
def insert_values(table_name,values,sql):
    """ insert a new vendor into the vendors table """
    conn = None
    vendor_id = None
    print(sql)
    try:
        # read database configuration
        # params = config()
        # connect to the PostgreSQL database
        conn = connect_postgress()
        #conn = psycopg2.connect(**params)
        # create a new cursor
        cur = conn.cursor()
        # execute the INSERT statement
        # cur.execute(sql, (values,))
        # execute the SQL statement
        execute_values(cur, sql, values)

        # get the generated id back
        #cur.fetchone()
        # cur.fetchall()
        # commit the changes to the database
        conn.commit()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        id_values = None
    finally:
        if conn is not None:
            conn.close()
            
def get_data(table_name):
    """ query data from the cell tables table """
    conn = None
    try:
        #params = config()
        conn = connect_postgress()
        cur = conn.cursor()
        cur.execute("SELECT id,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,stroke FROM "+table_name+" ORDER BY id")
        rowcount = cur.rowcount
        row = cur.fetchall()
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return row,rowcount
    
work_type_dict = {"Private":0, 
                  "Self-employed":1, 
                  "children":2, 
                  "Govt_job":3,
                  "Never_worked":4}
Residence_type_dict = {"Urban":0, 
                  "Rural":1}
smoking_status_dict = {"never smoked":0, 
                  "Unknown":1, 
                  "formerly smoked":2, 
                  "Govt_job":3,
                  "smokes":4}

def data_to_df(rows):
    df                 = pd.DataFrame(rows,columns=["id","gender","age","hypertension","heart_disease","ever_married","work_type","Residence_type","avg_glucose_level","bmi","smoking_status","stroke"]).set_index("id")
    return df

def change_on_dict(value,dict):
    return dict[value]

def preprocess(df_in):
    df                   = df_in[df_in["gender"] != "Other"].copy()
    df["bmi"]            = df["bmi"].fillna(df["bmi"].mean())
    df["ever_married"]   = np.where(df["ever_married"] == "Yes",1,0)
    df["ever_married"]   = df["ever_married"].astype("int")
    df["gender"]         = np.where(df["gender"] == "Female",1,0)
    df["gender"]         = df["gender"].astype("int")
    df = pd.get_dummies(df,columns=["work_type","Residence_type","smoking_status"])
    return df
    
def train_test(stroke):
    train_ds, test_ds = train_test_split(stroke, test_size=0.2)
    return train_ds, test_ds

def get_best_model(train_ds):
    label="stroke"
    predictor = task(label=label).fit(train_ds)
    model = predictor._trainer.load_model(predictor.get_model_best())
    print(model.get_info())
    return model, predictor
    
def eval_metrics(actual, pred):
        accuracy = accuracy_score(actual, pred)
        recall = recall_score(actual, pred, average='weighted')
        f1 = f1_score(actual, pred, average='weighted')
        return accuracy, recall, f1
        
def train_stroke(experiment_name):

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    
    mlflow.set_tracking_uri("http://nginx:80")
    mlflow.set_experiment(experiment_name)
    
    print(mlflow.get_tracking_uri())
    
    with mlflow.start_run():
        
        # Get data and train model with AutoML
        rows, rowcount = get_data("stroke")
        stroke         = data_to_df(rows)
        stroke         = preprocess(stroke)
        train_ds, test_ds  = train_test(stroke)
        model, predictor = get_best_model(train_ds)
        
        columns_test = list(test_ds.columns)
        columns_test.remove("stroke")
        test_ds_model = test_ds[columns_test]
        model.predict(test_ds_model)
        predicted_y = model.predict(test_ds_model)
        (accuracy, recall, f1) = eval_metrics(test_ds.stroke.values, predicted_y)

        print("  accuracy: %s" % accuracy)
        print("  recall: %s" % recall)
        print("  f1: %s" % f1)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model, "model", registered_model_name=experiment_name)
        else:
            mlflow.sklearn.log_model(model, "model")
            
def get_experiment_id(base_name,mlflow):
    n = 0
    experiment_id = 0
    while True:
        try:
            name = mlflow.get_experiment(n).name
        except:
            break
        print(name)
        if base_name == name:
            experiment_id = n
            break
        else:
            n = n + 1
    return experiment_id


def insert_data_batch(**context):

    # read Source data
    ds = read_data()
    print(ds)

    # try to create table if no exists:
    exist_tables = create_tables()
    
    # check inserted data
    insertion_step = 20
    row,rowcount = get_data("stroke")
    length = rowcount + insertion_step 
    
    # prepare data
    values = prepare_values(ds,rowcount,length)
    
    # insert data
    if values != "Full":
        print("WARNING: Data available to insert")
        table_name = "stroke"
        sql = """INSERT INTO """+table_name+""" (id,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,stroke) VALUES %s """
        insert_values(table_name,values,sql)
    else:
        print("WARNING: No more data to ingest")
    
def train_model_batch(**context):

    # Set experiment name
    experiment_name = 'stroke_demo_airflow'
    
    # Load data, train model using automl and register best model
    train_stroke(experiment_name)
    
    
def eval_model_batch(**context):

    mlflow.set_tracking_uri("http://nginx:80")
    model_name     = 'stroke_demo_airflow'

    # Import model on Staging
    try:
        stage  = "Staging"
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{stage}"
        )
        print("Model found on stagind and usign it")
    except:
        print("Not model found on staging. A MODEL SHOULD BE TAGGED AS STAGING")

    
    # Get data and train model with AutoML
    rows, rowcount = get_data("stroke")
    stroke         = data_to_df(rows)
    stroke         = preprocess(stroke)
    
    # evaluate new values
    train_ds, test_ds  = train_test(stroke)
    columns_test = list(test_ds.columns)
    columns_test.remove("stroke")
    test_ds_model = test_ds[columns_test]
    results = model.predict(test_ds_model)

    # Actual time
    now = datetime.now()
    print("now =", now)
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    
    # Generate results
    id_values = list(test_ds.index)
    df_results = pd.DataFrame(data=np.array([id_values,results]).T,columns=["id","stroke_pred"])
    df_results["timestamp"] = dt_string
    df_results = df_results[["id","timestamp","stroke_pred"]]
    df_results = df_results.set_index("id")

    # prepare data
    values = prepare_values(df_results,0,len(df_results))
    
    # Print results
    print(results)
    print(df_results)

    # Save results
    table_name = "results"
    sql = """INSERT INTO """+table_name+""" (id,timestamp,stroke_pred) VALUES %s """
    insert_values("stroke",values,sql)
