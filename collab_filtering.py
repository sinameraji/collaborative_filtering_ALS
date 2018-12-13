import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

#------------------
# LOAD THE DATASET
#------------------

data = pd.read_csv('data_cleaning/output/engagement.csv')

# Create a new dataframe without the user ids.
data_items = data.drop('user', 1)
