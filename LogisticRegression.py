import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imblearn

# import sqlalchemy
# from sqlalchemy.ext.automap import automap_base 
# from sqlalchemy.orm import Session 
# from sqlalchemy import create_engine, inspect, func
# from config import password
# engine =create_engine (f"postgresql://postgres:{password}@localhost:5432/Employee_Turnover")

# df = pd.read_sql_table('turnover_data',engine)

# DO THIS ONCE Split data set: majority of records for training and testing; putting some aside to see how well it predicts new data:
# df=pd.read_csv('Resources/turnoverData_full.csv')
# df_traintest = df[:1200]
# df_new = df[1200:]

# df_traintest.to_csv('Resources/train_test_data.csv', index=False)
# df_new.to_csv('Resources/prediction_data.csv', index=False)

df=pd.read_csv('Resources/train_test_data.csv')

df_skinny = df.drop(['EducationField','EmployeeCount','EmployeeNumber','JobLevel','StandardHours','JobRole','MaritalStatus','DailyRate','MonthlyRate','HourlyRate','Over18','OverTime'], axis=1).drop_duplicates()
df_skinny.rename(columns={"Attrition": "EmploymentStatus"}, inplace=True)

# Change qualitative data to numeric form
df_skinny['EmploymentStatus'] = df_skinny['EmploymentStatus'].replace(['Yes','No'],['Terminated','Retained'])
df_skinny['Gender']=df_skinny['Gender'].replace(['Female','Male'],[0,1])
df_skinny['BusinessTravel'] = df_skinny['BusinessTravel'].replace(['Travel_Rarely','Travel_Frequently','Non-Travel'],[1,2,0])
df_skinny['Department']=df_skinny['Department'].replace(['Human Resources','Sales','R&D'],[0,1,2])

# Note: 
# EmploymentStatus: 0=Active, 1=Terminated
# Gender: 0=female, 1=male
# Business Travel:  0=no travel, 1=rarely, 2=frequently
# Department: HR=0, Sales=1, R&D=2

import matplotlib.ticker as mtick

bars = ['Retained','Turnover']
y = df_skinny['EmploymentStatus'].value_counts()
y_as_percent = (y[0]/len(df_skinny),y[1]/len(df_skinny))

fig = plt.figure(1, (4.5,5))
ax = fig.add_subplot(1,1,1)

ax.bar(bars,y_as_percent, color=['Teal','Orange'])
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.xticks(fontsize=12)
plt.ylim(0,.95)
plt.yticks(fontsize=12)
plt.xlabel("\n Employment Status", fontsize=14)
plt.ylabel("Percent of Sample \n", fontsize=14)
plt.annotate("83.9%",xy=("Retained",.87),ha="center")
plt.annotate("16.1%",xy=("Turnover",.2),ha="center")
ax.tick_params(axis='both', which='major', pad=10)

plt.savefig('static/overallTurnover.png')

X =df_skinny.drop("EmploymentStatus", axis=1)
y = df_skinny["EmploymentStatus"]

# Data is imbalanced so need to resample.  The following (oversampling with a 60:40 ratio between 
# retained and terminated) gave results most consistent with dataset in sampling.
import imblearn
oversample = imblearn.over_sampling.RandomOverSampler(sampling_strategy=.4)
X_over, y_over = oversample.fit_resample(X, y)

from sklearn.model_selection import train_test_split
X_over_train, X_over_test, y_over_train, y_over_test = train_test_split(X_over, y_over, random_state=1)

from sklearn.preprocessing import StandardScaler
X_o_scaler = StandardScaler().fit(X_over_train)
X_o_train_scaled = X_o_scaler.transform(X_over_train)
X_o_test_scaled = X_o_scaler.transform(X_over_test)

from sklearn.linear_model import LogisticRegression
classifier_o = LogisticRegression()
classifier_o.fit(X_o_train_scaled, y_over_train)

# print(f"Training Data Score: {classifier_o.score(X_o_train_scaled, y_over_train)}")
# print(f"Testing Data Score: {classifier_o.score(X_o_test_scaled, y_over_test)}")

predictions = classifier_o.predict(X_o_test_scaled)
# print(f"First 10 Predictions:   {predictions[:10]}")
# print(f"First 10 Actual Employment Status: {y_over_test[:10].tolist()}")

# Predictions of new data
new_df = pd.read_csv("Resources/prediction_data.csv")
new_skinny = new_df.drop(['EducationField','EmployeeCount','EmployeeNumber','JobLevel','StandardHours','JobRole','MaritalStatus','DailyRate','MonthlyRate','HourlyRate','Over18','OverTime'], axis=1).drop_duplicates()
new_skinny.rename(columns={"Attrition": "EmploymentStatus"}, inplace=True)

new_skinny['EmploymentStatus'] = new_skinny['EmploymentStatus'].replace(['Yes','No'],['Terminated','Retained'])
new_skinny['Gender']=new_skinny['Gender'].replace(['Female','Male'],[0,1])
new_skinny['BusinessTravel'] = new_skinny['BusinessTravel'].replace(['Travel_Rarely','Travel_Frequently','Non-Travel'],[1,2,0])
new_skinny['Department']=new_skinny['Department'].replace(['Human Resources','Sales','R&D'],[0,1,2])

new_X = new_skinny.drop("EmploymentStatus", axis=1)
new_X_scaler = StandardScaler().fit(new_X)
new_X_scaled = new_X_scaler.transform(new_X)

new_o_predictions=classifier_o.predict(new_X_scaled)
# See how closely ratio of retained:terminated in predictions matches current data
# unique, counts = np.unique(new_o_predictions, return_counts=True)
# dict(zip(unique, counts))
# termpercent=((counts[1]/len(new_o_predictions))*100).round(1)
# print(dict(zip(unique,counts)))
# print(termpercent)

ynew = classifier_o.predict_proba(new_X_scaled)
# to get probability of termination of any given employee:
#probability = ynew[<index of ee's record>][1], # e.g:
# prob_ee1 = (ynew[7][1]*100).round(1)
# print(f"{prob_ee1}%")

columns = []
for col in df_skinny.drop('EmploymentStatus',axis=1).columns: 
    columns.append(col)

# Calculate weight of various factors on retention/turnover
feature_importance=pd.DataFrame(np.hstack((np.array([columns[0:]]).T, classifier_o.coef_.T)), columns=['feature', 'importance'])
feature_importance['importance']=pd.to_numeric(feature_importance['importance'])
plot_df=feature_importance.sort_values(by='importance', ascending=True)


y = plot_df['importance']
bars = plot_df['feature']
ticks = [-.45,.45]
labels = ['Weighs Heaviest on Retention','Weighs Heaviest on Turnover']

plt.figure(figsize=(15.5,5))

plt.barh(bars,y, height=.5, color='teal')
plt.xticks(ticks,labels,fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(-1,22)
plt.title("\n Weight of Employment Factors\n",fontsize=16)

plt.savefig('static/featureImportance.png')

# Feature importance for R&D workers
df_RD = df_skinny.loc[df_skinny['Department'].isin([2])].drop(["Department"],axis=1)

X_RD =df_RD.drop("EmploymentStatus", axis=1)
y_RD = df_RD["EmploymentStatus"]

oversample = imblearn.over_sampling.RandomOverSampler(sampling_strategy=.4)
X_RDover, y_RDover = oversample.fit_resample(X_RD, y_RD)

X_RD_train, X_RD_test, y_RD_train, y_RD_test = train_test_split(X_RDover, y_RDover, random_state=1)

X_RD_scaler = StandardScaler().fit(X_RD_train)
X_RD_train_scaled = X_RD_scaler.transform(X_RD_train)
X_RD_test_scaled = X_RD_scaler.transform(X_RD_test)

classifier=LogisticRegression()
classifier.fit(X_RD_train_scaled, y_RD_train)

columns_RD = []
for col in df_RD.drop('EmploymentStatus',axis=1).columns: 
    columns_RD.append(col)

feature_importance_RD=pd.DataFrame(np.hstack((np.array([columns_RD[0:]]).T, classifier.coef_.T)), columns=['feature', 'importance'])

feature_importance_RD['importance']=pd.to_numeric(feature_importance_RD['importance'])
plot_df_RD=feature_importance_RD.sort_values(by='importance', ascending=True)

y=plot_df_RD['importance']
bars=plot_df_RD['feature']
ticks = [-.6,.5]
labels = ['Weighs Heaviest on Retention','Weighs Heaviest on Turnover']

plt.figure(figsize=(15.6,5))
plt.barh(bars,y, height=.7, color='orange')
plt.xticks(ticks,labels,fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(-1,21)
plt.title("\n Weight of Employment Factors in R&D\n",fontsize=16)

plt.savefig('static/featureImportance_R&D.png')

# Feature importance for Sales workers
df_Sales = df_skinny.loc[df_skinny['Department'].isin([1])].drop(["Department"],axis=1)

X_S =df_Sales.drop("EmploymentStatus", axis=1)
y_S = df_Sales["EmploymentStatus"]

oversample = imblearn.over_sampling.RandomOverSampler(sampling_strategy=.4)
X_Sover, y_Sover = oversample.fit_resample(X_S, y_S)

X_S_train, X_S_test, y_S_train, y_S_test = train_test_split(X_Sover, y_Sover, random_state=1)

X_S_scaler = StandardScaler().fit(X_S_train)
X_S_train_scaled = X_S_scaler.transform(X_S_train)
X_S_test_scaled = X_S_scaler.transform(X_S_test)

classifier=LogisticRegression()
classifier.fit(X_S_train_scaled, y_S_train)

# print(f"Training Data Score: {classifier.score(X_RD_train_scaled, y_RD_train)}")
# print(f"Testing Data Score: {classifier.score(X_RD_test_scaled, y_RD_test)}")

columns_S = []
for col in df_Sales.drop('EmploymentStatus',axis=1).columns: 
    columns_S.append(col)

feature_importance_S=pd.DataFrame(np.hstack((np.array([columns_S[0:]]).T, classifier.coef_.T)), columns=['feature', 'importance'])

feature_importance_S['importance']=pd.to_numeric(feature_importance_S['importance'])
plot_df_S=feature_importance_S.sort_values(by='importance', ascending=True)

y=plot_df_S['importance']
bars=plot_df_S['feature']
ticks = [-.45,.55]
labels = ['Weighs Heaviest on Retention','Weighs Heaviest on Turnover']

plt.figure(figsize=(15.6,5))
plt.barh(bars,y, height=.7, color='red')
plt.xticks(ticks,labels,fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(-1,21)
plt.title("\n Weight of Employment Factors in Sales Department\n",fontsize=16)

plt.savefig('static/featureImportance_S.png')

