import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

df = pd.read_csv("C:/Users/ragha/Desktop/titanic.csv")
print(df.shape)
df.info()

print(df.head()) #First rows
print(df.tail()) #last rows

print(df.query("survived==1"))

print(df[["sex","fare","survived"]])

#Add column
new_column =df["class"] +" "+ df["embark_town"]
df["new_column"]= new_column
print(df)


#Remove column
del df["new_column"]
print(df)

#Filter _Males only
new= df["sex"] + df["survived"].astype(str)
df["new"]= new
print("number of males who survived:")
print(df["new"].value_counts()["male1"])

print("First ")
print(df["class"].value_counts()["First"])

print("Second ")
print(df["class"].value_counts()["Second"])

print("Third ")
print(df["class"].value_counts()["Third"])


print(sns.barplot(x='sex', y='survived',data=df))
plt.show()

sns.catplot(
    x="survived",
    hue ="embarked",
    kind ="count",
    col="embarked",
    data=df
)
plt.show()

sns.boxplot(x= "sex" , y ="age" , hue= "pclass"  , data=df )
plt.show()

sns.heatmap(df[["survived" , "age" ,"sibsp", "parch" , "fare"]].corr() ,
            vmin="0" , vmax="1")
plt.show()

sns.factorplot(x="sibsp" , y="survived" , data=df)
plt.show()


#################################################
sns.kdeplot(x="age" , y="fare" ,data=df, hue= "pclass" ,thresh = 0.1)
plt.show()

