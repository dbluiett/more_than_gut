import pandas as pd
import numpy as np
from customer_preds import get_data
import pickle

#Model relevant columns
c0= ["poutcome_failure", "poutcome_nonexistent","poutcome_success","month_oct","month_sep",
                    "month_mar","month_may","month_dec","month_apr", "education_university.degree","education_basic.4y",
                    "job_blue-collar", "job_admin.", "duration","campaign", "pdays","emp.var.rate", "cons.price.idx",
                    "euribor3m","nr.employed","cell_phone"]

c1 = ["age","duration","campaign", "pdays","emp.var.rate", "cons.price.idx", "cons.conf.idx",
                    "euribor3m","nr.employed","cell_phone", "clust_dist", "young_cust", "job_student", "marital_divorced",
                    "marital_single", "marital_married", "education_basic.9y","education_unknown","month_apr","month_dec",
                    "month_jul", "month_may","month_mar","month_oct","month_sep","poutcome_success", "poutcome_nonexistent"]

c2 = ["duration","campaign", "pdays","emp.var.rate", "cons.price.idx", "cons.conf.idx",
                    "euribor3m","nr.employed","cell_phone", "job_admin.","job_blue-collar",
                     "education_university.degree","month_dec",
                     "month_may","month_mar","month_oct","month_sep","poutcome_success", "poutcome_nonexistent"]

c3 = ["age", "duration","campaign", "pdays","emp.var.rate", "cons.price.idx", "cons.conf.idx",
                    "euribor3m","nr.employed","cell_phone","clust_dist","mature_cust", "job_blue-collar",
                     "education_basic.4y","month_dec", "job_retired", "job_self-employed", "job_services",
                     "job_technician", "education_basic.6y", "month_apr","month_jul","month_jun","poutcome_failure",
                     "month_may","month_mar","month_oct","month_sep","poutcome_success", "poutcome_nonexistent"]

#Unpickling models
file=open("data/persona0_predictor.pkl",'r')
model0= pickle.load(file)
file=open("data/persona1_predictor.pkl",'r')
model1 =pickle.load(file)
file=open("data/persona2_predictor.pkl",'r')
model2=pickle.load(file)
file=open("data/persona3_predictor.pkl",'r')
model3=pickle.load(file)

# My functions
def get_point():
    x = np.random.randint(df.shape[0])
    point = df.iloc[x]
    return pd.DataFrame(point).T

def fake_point(point):
    point["cell_phone"]=True
    point["cons.price"] = 93.576
    point["emp.var.rate"]=0.08
    point["euribor3m"]= 3.62
    point["nr.employed"] = 5167
    return test_point(point)


def test_point (point):
    for i,x in enumerate(clusters_dict.values()):
        if point.age.unique()[0] in x:
            cluster_name = clusters_dict.keys()[i]
            cluster = clusters_int[cluster_name]
    columns = ["education", "job","marital","month"]
    transform_full= pd.get_dummies(point, columns = columns)
    if point.age.unique()[0]>65:
        transform_full["mature_cust"]=True
    if point.age.unique()[0]<26:
        transform_full["young_cust"]=True
    missing=pd.DataFrame(columns=np.setdiff1d(cluster, transform_full.columns))
    new_point = pd.concat([transform_full,missing], axis=1).fillna("0")
    new_point=new_point[cluster]
    model = model_dict[cluster_name]
    prediction = model.predict(new_point)
    outcome ="na"
    if "target" in point.columns:
        outcome = point.target.unique()[0]
    #print "Persona: ",cluster_name," Prediction: ",prediction, "Actual: ", outcome
    return cluster_name, prediction, outcome

#Get a point
df = get_data()
my_point = get_point()

#Creating dictionaries for the personas
clusters_int = {"Jim":c0, "Joey":c1, "Jack":c2, "John":c3}
clusters_dict = {"Jim":range(44,53), "Joey":range(18,34), "Jack":range(34,44), "John":range(53,100)}
model_dict ={"Jim":model0, "Joey":model1, "Jack":model2, "John":model3}
