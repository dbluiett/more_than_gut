import pandas as pd
import numpy as np 

from prediction2 import create_model, evaluate_model
from sklearn.cross_validation import train_test_split 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV

def get_data():
    df = pd.read_csv("data/clean_data.csv")
    df["young_cust"] = df.age<26
    df["mature_cust"] = df.age>60
    df["target"]= df.y=="yes"
    df.drop("y", axis=1, inplace=True)
    return df

def transform_model(df):
    df_transform = pd.get_dummies(df)
    return df_transform

def test_model(df):
    gbc= GradientBoostingClassifier()
    test_data(df, gbc)
    gbc_params = {
    'min_samples_leaf': [1, 5, 10, 20],
    'n_estimators': [100,200,300,400],
    'learning_rate': [0.01, 0.025, 0.05, 0.1], 
    }
    search = GridSearchCV(gbc, gbc_params, n_jobs=-1, scoring = "f1")
    search.fit(X_train, y_train)
    preds_b = search.best_estimator_.predict(X_test)
    evaluate_model(preds_b, y_test)
    return search.best_estimator_

def test_data(df,model, fit=False):
    target = df.target
    features = df.drop("target", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features,target)
    if fit:
        preds =model.predict(X_test)
    else:
        preds,_=create_model(model,X_train, X_test, y_train)
    evaluate_model(preds,y_test)


# Get Data
df = get_data()
persona0= df[df.cluster==0]
persona1= df[df.cluster==1]
persona2= df[df.cluster==2]
persona3= df[df.cluster==3]

#Build subsets
P0=transform_model(persona0)
persona0_columns = ["poutcome_failure", "poutcome_nonexistent","poutcome_success","month_oct","month_sep",
                    "month_mar","month_may","month_dec","month_apr", "education_university.degree","education_basic.4y",
                    "job_blue-collar", "job_admin.", "duration","campaign", "pdays","emp.var.rate", "cons.price.idx",
                    "euribor3m","nr.employed","cell_phone","target"]

P0=P0[persona0_columns]
P1 = transform_model(persona1)
persona1_columns = ["age","duration","campaign", "pdays","emp.var.rate", "cons.price.idx", "cons.conf.idx",
                    "euribor3m","nr.employed","cell_phone", "clust_dist", "young_cust", "job_student", "marital_divorced",
                    "marital_single", "marital_married", "education_basic.9y","education_unknown","month_apr","month_dec",
                    "month_jul", "month_may","month_mar","month_oct","month_sep","poutcome_success", "poutcome_nonexistent","target"]

P1=P1[persona1_columns]

P2 = transform_model(persona2)
persona2_columns = ["duration","campaign", "pdays","emp.var.rate", "cons.price.idx", "cons.conf.idx",
                    "euribor3m","nr.employed","cell_phone", "job_admin.","job_blue-collar",
                     "education_university.degree","month_dec",
                     "month_may","month_mar","month_oct","month_sep","poutcome_success", "poutcome_nonexistent","target"]

P2=P2[persona2_columns]
P3 = transform_model(persona3)
persona3_columns = ["age", "duration","campaign", "pdays","emp.var.rate", "cons.price.idx", "cons.conf.idx",
                    "euribor3m","nr.employed","cell_phone","clust_dist","mature_cust", "job_blue-collar",
                     "education_basic.4y","month_dec", "job_retired", "job_self-employed", "job_services",
                     "job_technician", "education_basic.6y", "month_apr","month_jul","month_jun","poutcome_failure",
                     "month_may","month_mar","month_oct","month_sep","poutcome_success", "poutcome_nonexistent","target"]

P3=P3[persona3_columns]










