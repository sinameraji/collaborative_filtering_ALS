import pandas as pd
import re
courses = pd.read_csv("output/courses.csv", header=1)

courses.columns = ["id", "title", "url", "type", "price"]

id_list = []
count = 10000
for c in courses.iloc[:,1]:
    id = str(count) + ""
    for w in c.split(" "):
        # print (w)
        if(w != ""):
            w = re.sub('[^A-Za-z0-9]+', '', w)
            id = id + w[0]
    id_list.append(id)
    count = count + 1

courses["id"] = id_list
# print (courses.iloc[0,:])


courses.to_csv("output/cleaned.csv", sep=',', index=False)