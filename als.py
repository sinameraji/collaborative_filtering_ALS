import random
import pandas as pd
import numpy as np

import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler

#-------------------------
# LOAD AND PREP THE DATA
#-------------------------
 
raw_data = pd.read_csv('data_cleaning/output/engagement.csv', header=0)
raw_data.columns = ['student', 'course', 'engagement']
 
#  Drop rows with missing values
data = raw_data.dropna()

# Convert courses' names into numerical IDs
data['studentid'] = data['student'].astype("category").cat.codes
data['courseid'] = data['course'].astype("category").cat.codes
print(data)

# Create a lookup frame so we can get the course names back in 
 # readable form later.
item_lookup = data[['courseid', 'course']].drop_duplicates()
item_lookup['courseid'] = item_lookup.courseid.astype(str)

data = data.drop(['student', 'course'], axis=1)

# Create lists of all students, courses and engagements
students = list(np.sort(data.studentid.unique()))
courses = list(np.sort(data.courseid.unique()))
engagements = list(data.engagement)


# Get the rows and columns for our new matrix
rows = data.studentid.astype(int)
cols = data.courseid.astype(int)

# Contruct a sparse matrix for our students and courses containing engagements
data_sparse = sparse.csr_matrix((engagements, (rows, cols)), shape=(len(students), len(courses)))
# print(data_sparse)


def nonzeros(m, row):
    for index in range(m.indptr[row], m.indptr[row+1]):
        yield m.indices[index], m.data[index]
      
      
def implicit_als_cg(Cui, features=20, iterations=20, lambda_val=0.1):
    user_size, item_size = Cui.shape

    X = np.random.rand(user_size, features) * 0.01
    Y = np.random.rand(item_size, features) * 0.01

    Cui, Ciu = Cui.tocsr(), Cui.T.tocsr()

    for iteration in range(iterations):
        print ('iteration %d of %d' % (iteration+1, iterations))
        least_squares_cg(Cui, X, Y, lambda_val)
        least_squares_cg(Ciu, Y, X, lambda_val)
    
    return sparse.csr_matrix(X), sparse.csr_matrix(Y)
  
  
def least_squares_cg(Cui, X, Y, lambda_val, cg_steps=3):
    users, features = X.shape
    
    YtY = Y.T.dot(Y) + lambda_val * np.eye(features)

    for u in range(users):

        x = X[u]
        r = -YtY.dot(x)

        for i, confidence in nonzeros(Cui, u):
            r += (confidence - (confidence - 1) * Y[i].dot(x)) * Y[i]

        p = r.copy()
        rsold = r.dot(r)

        for it in range(cg_steps):
            Ap = YtY.dot(p)
            for i, confidence in nonzeros(Cui, u):
                Ap += (confidence - 1) * Y[i].dot(p) * Y[i]

            alpha = rsold / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap

            rsnew = r.dot(r)
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        X[u] = x

alpha_val = 15
conf_data = (data_sparse * alpha_val).astype('double')
user_vecs, item_vecs = implicit_als_cg(conf_data, iterations=50, features=20)

# print(user_vecs)

#------------------------------
# FIND SIMILAR ITEMS
#------------------------------

# Let's find similar courses to Coursera ML
# Note that this ID might be different for you if you're using
# the full dataset or if you've sliced it somehow. 
# print(item_lookup)
item_id = 0

# Get the item row for Coursera ML
item_vec = item_vecs[item_id].T
# print(item_vec)

# Calculate the similarity score between Coursera ML and other courses
# and select the top 10 most similar.
scores = item_vecs.dot(item_vec).toarray().reshape(1,-1)[0]
top_10 = np.argsort(scores)[::-1][:10]

courses = []
course_scores = []

# Get and print the actual course names and scores
for idx in top_10:
    courses.append(item_lookup.course.loc[item_lookup.courseid == str(idx)].iloc[0])
    course_scores.append(scores[idx])

similar = pd.DataFrame({'course': courses, 'score': course_scores})

print (similar)

# recommend courses for kamwoh (user with ID 3)
user_id = 1

#------------------------------
# GET ITEMS CONSUMED BY USER
#------------------------------

# Let's print out what courses user has studied 
consumed_idx = data_sparse[user_id,:].nonzero()[1].astype(str)
consumed_items = item_lookup.loc[item_lookup.courseid.isin(consumed_idx)]
print (consumed_items)


#------------------------------
# CREATE USER RECOMMENDATIONS
#------------------------------

def recommend(user_id, data_sparse, user_vecs, item_vecs, item_lookup, num_items=10):
    """Recommend items for a given user given a trained model
    
    Args:
        user_id (int): The id of the user we want to create recommendations for.
        
        data_sparse (csr_matrix): Our original training data.
        
        user_vecs (csr_matrix): The trained user x features vectors
        
        item_vecs (csr_matrix): The trained item x features vectors
        
        item_lookup (pandas.DataFrame): Used to map course ids to course names
        
        num_items (int): How many recommendations we want to return:
        
    Returns:
        recommendations (pandas.DataFrame): DataFrame with num_items course names and scores
    
    """
  
    # Get all interactions by the user
    user_interactions = data_sparse[user_id,:].toarray()

    # We don't want to recommend items the user has consumed. So let's
    # set them all to 0 and the unknowns to 1.
    user_interactions = user_interactions.reshape(-1) + 1 #Reshape to turn into 1D array
    user_interactions[user_interactions > 1] = 0

    # This is where we calculate the recommendation by taking the 
    # dot-product of the user vectors with the item vectors.
    rec_vector = user_vecs[user_id,:].dot(item_vecs.T).toarray()

    # Let's scale our scores between 0 and 1 to make it all easier to interpret.
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
    recommend_vector = user_interactions*rec_vector_scaled
   
    # Get all the artist indices in order of recommendations (descending) and
    # select only the top "num_items" items. 
    item_idx = np.argsort(recommend_vector)[::-1][:num_items]

    courses = []
    scores = []

    # Loop through our recommended artist indicies and look up the actial artist name
    for idx in item_idx:
        courses.append(item_lookup.course.loc[item_lookup.courseid == str(idx)].iloc[0])
        scores.append(recommend_vector[idx])

    # Create a new dataframe with recommended artist names and scores
    recommendations = pd.DataFrame({'course': courses, 'score': scores})
    
    return recommendations

# Let's generate and print our recommendations
recommendations = recommend(user_id, data_sparse, user_vecs, item_vecs, item_lookup)
print (recommendations)