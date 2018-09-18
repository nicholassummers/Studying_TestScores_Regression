# coding: utf-8

# In[50]:
#URL to get data: https://archive.ics.uci.edu/ml/datasets/Student+Performance#
#import necessary libraries
import pandas as pd #for data loading and descriptive statistics
import statsmodels.api as sm
import numpy as np
#read data file
data = pd.read_csv("pathtocsvfile.csv")
#look into dataset: head looks at first instances and tail look at the last
#the parameters specifies the number of instances to look at
data.head(5)
data.tail(5)
#covert csv data to pandas dataframe (contains rows and columns)
dataFrame = pd.DataFrame(data)
#save data frame as a file type, file's name is parameter
dataFrame.to_csv("studentDataFrame.csv")
#calculate descriptive statistics
avg = dataFrame.mean()
max = dataFrame.max()
#see which columns are in our data
#list(dataFrame)
#fetch specific columns to include in regression analysis
studytime = dataFrame[dataFrame.columns[13:14]]
grades1 = dataFrame[dataFrame.columns[30:31]]
grades2 = dataFrame[dataFrame.columns[31:32]]
grades3 = dataFrame[dataFrame.columns[32:33]]
#print descriptive statistics to the console
average = dataFrame.mean() #prints average of each column
#print(average)
results1 = sm.OLS(grades1, studytime).fit()
results2 = sm.OLS(grades2, studytime).fit()
results3 = sm.OLS(grades3, studytime).fit()
print(results1.summary())
print(results2.summary())
print(results3.summary())
#use linear regression to predict test scores using studytime as independent variable
print("If you study for 4 hours, your test score out of 20 will be ", results1.predict(4))
