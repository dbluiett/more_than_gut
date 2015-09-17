import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_csv("/Users/datascientist/Downloads/crime.csv")
columns = ['GEO_LON','GEO_LAT','PRECINCT_ID','NEIGHBORHOOD_ID']

df=data[columns]
crimes = ['aggravated-assault','other-crimes-against-persons','robbery','murder','sexual-assault']
danger_zone=data.OFFENSE_CATEGORY_ID.apply(lambda x: x in crimes)

#Cleaning data
time= [x.to_datetime().hour for x in pd.to_datetime(data.FIRST_OCCURRENCE_DATE)]
df["time_of_day"]=time
df["danger_zone"]=danger_zone.copy()

#df_with_dummies = pd.get_dummies(df)
#Test/Training data

X_train, X_test, Y_train, Y_test = train_test_split(df,danger_zone, random_state=84)
gbc=GradientBoostingClassifier(verbose=1, warm_start=True)
gbc.fit(X_train, Y_train)
pred_proba = gbc.predict_proba(X_test)
predictions = gbc.predict(X_test)
accuracy = metrics.accuracy_score(Y_test,predictions)
F1= metrics.f1_score(Y_test, predictions)
print "accuracy:", accuracy
print "F1:", F1

