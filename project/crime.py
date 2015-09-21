import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

data = pd.read_csv("./data/crime.csv")

top_30 =["five-points", "cbd","stapleton","capitol-hill","montbello","baker", "lincoln-park", "westwood", "east-colfax",
        "union-station","gateway-green-valley-ranch","civic-center","highland","west-colfax","north-capitol-hill",
        "speer","hampden-south","hampden","northeast-park-hill","villa-park","washington-virginia-vale","virginia-village","city-park-west",
        "college-view-south-platte","cheesman-park","mar-lee","sunnyside","athmar-park","cole","goldsmith"]

columns = ['GEO_LON','GEO_LAT','PRECINCT_ID']

df=data[columns]
crimes = ['aggravated-assault','other-crimes-against-persons','robbery','murder','sexual-assault']
df["target"]=data.OFFENSE_CATEGORY_ID.apply(lambda x: x in crimes)

#Cleaning data
time= [x.to_datetime().hour for x in pd.to_datetime(data.FIRST_OCCURRENCE_DATE)]
df["time_of_day"]=time

my_neighbors=data.NEIGHBORHOOD_ID.apply(lambda x: x if x in top_30 else "other")
df["area"]=my_neighbors.copy()
df.dropna(inplace=True)
danger_zone=df.pop("target")

df_with_dummies = pd.get_dummies(df, columns=["area"])

#Test/Training data

X_train, X_test, Y_train, Y_test = train_test_split(df_with_dummies,danger_zone, random_state=84)
gbc=GradientBoostingClassifier(verbose=1, warm_start=True)
gbc.fit(X_train, Y_train)
pred_proba = gbc.predict_proba(X_test)

#Adjusting the probability cutoff 
new_proba = pred_proba[:,1]
predictions = [x>0.30 for x in new_proba]
accuracy = metrics.accuracy_score(Y_test,predictions)
F1= metrics.f1_score(Y_test, predictions)

""" recall = 87%, accuracy = 37%, cutoff is 0.05; recall = 55%, accuracy = 70%, cutoff =0.08; recall 41%, accuracy = 80%, cutoff is 0.1"""
