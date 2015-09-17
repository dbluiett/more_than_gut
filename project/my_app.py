from flask import Flask
from flask import request, jsonify, render_template, make_response
import pandas as pd
import json
import sys
import glob
import numpy as np
import argparse

from pyxley import UILayout
from pyxley.filters import SelectButton
from pyxley.charts.mg import LineChart, Figure, ScatterPlot, Histogram
from pyxley.charts.datatables import DataTable
from collections import OrderedDict

from personas import Feature_corr_dict, Plot_feature_weights, persona_dict, model_txt_dict
from customer_preds import get_data
from get_point import get_point, test_point, fake_point


parser = argparse.ArgumentParser(description="Flask Template")
parser.add_argument("--env", help="production or local", default="local")
args = parser.parse_args()

TITLE = "More than Gut!"

scripts = [
    "./bower_components/jquery/dist/jquery.min.js",
    "./bower_components/datatables/media/js/jquery.dataTables.js",
    "./bower_components/d3/d3.min.js",
    "./bower_components/metrics-graphics/dist/metricsgraphics.js",
    "./bower_components/require/build/require.min.js",
    "./bower_components/react/react.js",
    "./bower_components/react-bootstrap/react-bootstrap.min.js",
    "./bower_components/pyxley/build/pyxley.js",
]

css = [
    "./bower_components/bootstrap/dist/css/bootstrap.min.css",
    "./bower_components/metrics-graphics/dist/metricsgraphics.css",
    "./bower_components/datatables/media/css/jquery.dataTables.min.css",
    "./css/main.css"
]

# Make a UI: the url is embedded in the "./static/bower_components..."
ui = UILayout(
    "FilterChart",
    "./static/bower_components/pyxley/build/pyxley.js",
    "component_id",
    filter_style="''")

#importing the raw data
data = get_data()
df1 = data[["target","default","housing","age","education"]]
df_age = data[["cluster", "age"]]
names= ["Jim", "Joey", "Jack", "John"]
df_age["persona"] = df_age.cluster.apply(lambda x: names[x])

cols2 = OrderedDict([
    ("target", {"label": "Purchased"}),
    ("default", {"label": "In Default?"}),
    ("housing", {"label": "Mortgage"}),
    ("age", {"label": "Age"}),
    ("education",{"label":"Education"})
])


# Make a Button
btn = SelectButton("Persona", names, "persona", "Jim")

# Now make a FilterFrame for the histogram
hFig = Figure("/mghist/", "myhist")
hFig.layout.set_size(width=350, height=200)
hFig.layout.set_margin(left=0, right=0)
hFig.graphics.animate_on_load()

# Make a histogram 
hc2 = Histogram(df_age, hFig, "age", 5, title="Customer Persona Age", init_params={"persona":"John"})


# Adding button and histogram to UI

ui.add_chart(hc2)
ui.add_filter(btn)

app = Flask(__name__)
sb = ui.render_layout(app, "./static/layout.js")


@app.route('/', methods=["GET"])

@app.route('/home', methods=["GET"])
def home():
    return render_template("my_home.html")

@app.route('/point', methods=["GET"])
def point():
    return render_template("get_point.html", title=TITLE)


@app.route('/predict', methods =["GET","POST"])
def predict():
    point = get_point()
    cluster, prediction, actual = test_point(point)
    return render_template('predict_template.html', title = TITLE, cluster = cluster, prediction=prediction, actual = actual)
    #'Persona: %s, <br> Prediction: %s, <br> Actual: %s' % (cluster, prediction, actual)

@app.route('/predict2', methods =["GET","POST"])
def predict2():
    point_df = pd.DataFrame(columns=["age", "marital", "education", "job", "campaign", "p_outcome", "duration", "p_days","month"])
    data=[]
    data.append(int(request.form["age"]))
    data.append(request.form["marital"])
    data.append(request.form["education"])
    data.append(request.form["job"])
    data.append(int(request.form["campaign"]))
    data.append(request.form["p_outcome"])
    data.append(int(request.form["duration"]))
    data.append(int(request.form["p_days"]))
    data.append(request.form["month"])
    point_df.loc[0]=data
    cluster, prediction, actual = fake_point(point_df)
    return render_template('predict_template.html', title = TITLE, cluster = cluster, prediction=prediction, actual = actual)


@app.route('/description', methods =["GET","POST"])
def description():
    cluster = request.form["persona_name"]
    weights = Feature_corr_dict[cluster]
    Plot_feature_weights(weights)
    desc = persona_dict[cluster]
    model = model_txt_dict[cluster]
    return render_template('insight.html', title = TITLE, cluster = cluster, desc = desc, model = model)

@app.route('/dashboard', methods=["GET","POST"])
def index():
    _scripts = ["./layout.js"]
    return render_template('test.html',
        title=TITLE,
        base_scripts=scripts,
        page_scripts=_scripts,
        css=css)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=80)
