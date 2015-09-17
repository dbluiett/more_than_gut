import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import sklearn.metrics as metrics
from sklearn.grid_search import GridSearchCV



def create_model(model, trainx, testx, trainy):
    model.fit(trainx, trainy)
    preds = model.predict(testx)
    probability = model.predict_proba(testx)
    return preds, probability

def evaluate_model(preds, testy):
    accuracy = metrics.accuracy_score(testy, preds)
    precision = metrics.precision_score(testy, preds)
    recall = metrics.recall_score(testy, preds)
    F1 = metrics.f1_score(testy, preds)
    Fbeta = metrics.fbeta_score(testy, preds, 2) # weighting recall stronger than precision
    print "Model summary: accuracy - ", accuracy,"precision - ",precision, "recall - ", recall, "Fbeta - ",Fbeta, "F1 - ",F1

def optimize_model(model, params, trainx, testx, trainy):
    search = GridSearchCV(model, params, n_jobs=-1, scoring='f1')
    search.fit(trainx, trainy)
    preds = search.best_estimator_.predict(testx)
    print "Best model parameters: ", search.best_params_
    return preds, search.best_params_

corr_features = ["poutcome_nonexistent", "poutcome_success", "education_university.degree",
                    "education_basic.9y", "marital_married","marital_single", "job_student",
                    "job_retired", "job_blue-collar", "duration", "campaign", "pdays","previous",
                    "emp.var.rate", "cons.price.idx","cons.conf.idx", "euribor3m", "nr.employed", "cell_phone",
                    "month_oct", "month_dec", "month_mar", "month_apr", "month_may", "month_sep"]
