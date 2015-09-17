import pandas as pd 
import numpy as np 
from customer_preds import P0, P1, P2, P3
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import StringIO

# My functions 
def Plot_feature_weights(feature_corr):
    fig, axes = plt.subplots(nrows=1, ncols=1)
    if "target" in feature_corr.index:
        feature_corr.drop("target", inplace = True)
    feature_corr.plot(kind = "barh",ax=plt.gca(), grid = False, fontsize=8)
    plt.gcf().subplots_adjust(left=.2)
    plt.savefig("static/corr_weight.jpg")
    #import pdb; pdb.set_trace()
    canvas=FigureCanvas(fig)
    png_output = StringIO.StringIO()
    canvas.print_png(png_output)
    return png_output

#Calculate feature correlation to target
Jim_corr = P0.corrwith(P0.target, axis=0)
Joey_corr = P1.corrwith(P1.target, axis=0)
Jack_corr = P2.corrwith(P2.target, axis=0)
John_corr = P3.corrwith(P3.target, axis=0)

#Create dictionary of feature_correlations
Feature_corr_dict ={"Jim":Jim_corr, "Joey": Joey_corr, "Jack": Jack_corr, "John": John_corr}

#Create persona descriptions
John_description = "This customer group represents 10% of the total customers and 19% of the customers who purchase this product. 1 in 10 of the customers in their 50's are likely to purchase this product. 1 in 2 of the customers over 60 are likely to purchase, however they only represent 16% of this group and 2% of the overall customer population"
Jim_description = "This customer group represents 25% of the total customers.  1 in 10 are likely to purchase this product, representing 18% of the customers who purchased."
Joey_description = "This customer group represents 30% of the total customers and 33% of the customers who purchased this product.  1 in 10 of the customers aged 26-32 are likely to purchase this product.  1 in 5 of the customers under 26 are likely to purchase, however they only represent 13% of this group and 4% of the overall customer population."
Jack_description = "This customer group represents 33% of the total customers. 1 in 10 are likely to purchase this product, representing 30% of the customers who purchaseed."

#Creating model descriptions
Jim_model = "Model summary: Accuracy -  0.95, Precision -  0.72, recall -  0.54, F1 -  0.62"
John_model = "Model summary: Accuracy -  0.90, Precision -  0.69, recall -  0.63, F1 -  0.66"
Jack_model = "Model summary: Accuracy -  0.94, Precision -  0.78, recall -  0.55, F1 -  0.65"
Joey_model = "Model summary: Accuracy -  0.92, Precision -  0.75, recall -  0.60, F1 -  0.67"

#Create dictionary for descriptions
persona_dict ={"Jim":Jim_description, "Joey": Joey_description, "Jack": Jack_description, "John": John_description}
model_txt_dict ={"Jim":Jim_model, "Joey": Joey_model, "Jack": Jack_model, "John": John_model}
