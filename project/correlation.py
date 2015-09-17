import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt 

#Look at correllation matrix
def Show_correlation(df, cols=None):
    """
    Use a heatmap of the correlations of DataFrame columns to estimate the
    features to engineer.
    """
    if cols:
        df = df[cols + ['target']]
    corrmat = df.corr()
    plt.figure(figsize = (12, 10))
    cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)
    sns.heatmap(corrmat, cmap=cmap, annot=True, fmt="f")
    plt.xticks(rotation = 90); plt.yticks(rotation = 0)
    plt.tight_layout()
    plt.show()