import pandas as pd

df = pd.read_csv("output/cleaned.csv", header=1)
df.columns = ["id", "title", "url", "type", "price"]

# table
# new_columns = ['student', 'course', 'engagement']
# new_df  = pd.DataFrame(columns = new_columns)
# new_df["course"] = df.iloc[:,0]
# new_df["engagement"] = 0
# new_df["student"] = "sina"
# print(new_df)


# user-item
new_df  = pd.DataFrame(columns = ["student",df["id"]])
new_df.iloc[0,1:] = 0
print(new_df)



# new_df.to_csv("output/engagement.csv", sep=',', index=False)