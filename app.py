import pandas as pd
import os
# from config import password
import psycopg2
import sqlalchemy
import json 
import flask
import numpy as np
import LogisticRegression
import scrape
password = 'Parvin123!!'
from sqlalchemy import create_engine

DATABASE_URL = os.environ['DATABASE_URL']

from flask import Flask, render_template, redirect, jsonify

app=Flask(__name__)

# Run scrape function to get latest national data from BLS website and load it into database
scrape.scrape_bls()

# Draw employee turnover data and national statistics from database
connection = psycopg2.connect(DATABASE_URL, sslmode='require')
postgreSQLTable = "blsdata"

initial_df=pd.read_sql("select * from turnover_data",connection)
# alternative: read data from csv file:
# initial_df=pd.read_csv("Resources/turnoverData_full.csv")

bls_df=pd.read_sql("select * from blsdata", connection)
connection.close()

bls_df.reset_index(drop=True, inplace=True)
bls_df=bls_df.drop(bls_df.index[[35,36]])

bls_html_table = bls_df.to_html()

# Clean employee data: remove less helpful columns; rename values to be more user-friendly
df = initial_df.drop(['EducationField','EmployeeCount','Education','StandardHours','JobRole','MaritalStatus',
                      'DailyRate','MonthlyRate','HourlyRate','Over18','OverTime','TotalWorkingYears'], axis=1).drop_duplicates()
df.rename(columns={'Attrition': 'Employment Status','BusinessTravel':'Business Travel','DistanceFromHome':'Commute (Miles)','EnvironmentSatisfaction':'Environment Satisfaction',
                  'JobInvolvement':'Job Involvement','JobLevel':'Job Level','JobSatisfaction':'Job Satisfaction',
                  'MonthlyIncome':'Monthly Income','NumCompaniesWorked':'Num Companies Worked','PercentSalaryHike':'Last Increase %',
                  'PerformanceRating':'Performance Rating','RelationshipSatisfaction':'Relationship Satisfaction','StockOptionLevel':'Stock Option Level',
                  'TrainingTimesLastYear':'Training Last Year','WorkLifeBalance':'Work/Life Balance','YearsAtCompany':'Tenure (Years)',
                  'YearsInCurrentRole':'Years in Role','YearsSinceLastPromotion':'Years Since Promotion','YearsWithCurrManager':'Years with Manager'}, inplace = True)

df['Employment Status'] = df['Employment Status'].replace(['No','Yes'],['Active','Terminated'])
factors = list(df.drop("EmployeeNumber",axis=1))
 
# TRAIN PREDICTOR
    
df['Gender']=df['Gender'].replace(['Female','Male'],['0','1'])
df['Business Travel'] = df['Business Travel'].replace(['Travel_Rarely','Travel_Frequently','Non-Travel'],['1','2','0'])
df['Department']=df['Department'].replace(['Human Resources','Sales','R&D'],['0','1','2'])
X = df.drop("Employment Status", axis=1)
y = df["Employment Status"]
    
# Data is imbalanced so need to resample.  The following method (oversampling with a 60:40 ratio between 
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

# Go back to original full data set, minus the 237 who have already left the company, to make predictions
df_predict=df[df["Employment Status"]=="Active"]
X_predict = df_predict.drop("Employment Status", axis=1)
y_predict = df_predict["Employment Status"]
X_scaler=StandardScaler().fit(X_predict)
X_scaled=X_scaler.transform(X_predict)

predictions=classifier_o.predict(X_scaled)
prob = classifier_o.predict_proba(X_scaled)
# Convert prob from ndarray to python list
prob_lst = prob.tolist()

loss_probability = []
    
for i in prob_lst:
    index=prob_lst.index(i)
    probability = (prob_lst[index][1]*100)
    loss_probability.append(probability)

emp_nums = df['EmployeeNumber'].to_list()

# Serve html templates and data to site
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dataTools")
def dataTool():
    return render_template("dataTools.html")

@app.route("/deepDive")
def deepDive():
    return render_template("deepDive.html")

@app.route("/resources")
def resources():
    
    return render_template("resources.html",table=bls_html_table, id="blsTable")

@app.route("/resScientists")
def resources_slideshow():
    return render_template("scientistsQuit.html") 

@app.route("/millVideo")
def millVideo():
    return render_template("millenials_video.html")

@app.route("/turnover_cost")
def cost_calculation():
    df_level = df.groupby(['Job Level','Employment Status']).count().reset_index()
    
    lev1_turn = df_level.iloc[1,2].item()
    lev2_turn = df_level.iloc[3,2].item()
    lev3_turn = df_level.iloc[5,2].item()
    lev4_turn = df_level.iloc[7,2].item()
    lev5_turn = df_level.iloc[9,2].item()
    
    cost = [4000, 6000, 8000, 18000, 40000]
    avg_cost = (((cost[0]*lev1_turn) + (cost[1] * lev2_turn) + (cost[2] * lev3_turn) + (cost[3] * lev4_turn) + 
    (cost[4] * lev5_turn))/(lev1_turn + lev2_turn + lev3_turn + lev4_turn + lev5_turn))
    total_cost=int(avg_cost * (lev1_turn + lev2_turn + lev3_turn + lev4_turn + lev4_turn))
    total_count = len(df)

    y=df['Employment Status'].value_counts()
    retained_rate=(y[0]/len(df)*100).round(2)
    turnover_count=(y[1]).item()
    turnover_rate=((y[1]/len(df)*100)).round(2)
    
    proportion_turnover_lev1 = (lev1_turn/turnover_count)
    proportion_turnover_lev2 = (lev2_turn/turnover_count)
    proportion_turnover_lev3 = (lev3_turn/turnover_count)
    proportion_turnover_lev4 = (lev4_turn/turnover_count)
    proportion_turnover_lev5 = (lev5_turn/turnover_count)
    
    calc_dict = [{"total_employees":total_count}, {"cost_per_level":cost},{"avg_cost":avg_cost},{"total_cost":total_cost},
    {"turnover_count":turnover_count},{"turnover_rate":turnover_rate},{"retained_rate":retained_rate},{"proportion_lev1_turnover":proportion_turnover_lev1}, 
    {"proportion_lev2_turnover":proportion_turnover_lev2}, {"proportion_lev3_turnover":proportion_turnover_lev3},
    {"proportion_lev4_turnover":proportion_turnover_lev4}, {"proportion_lev5_turnover":proportion_turnover_lev5},]

    calc_data = json.loads(json.dumps(calc_dict))
     
    return jsonify(calc_data)    

@app.route("/employee_data")

def post_predictions():
    emp_dicts = []
    for i, j in zip(emp_nums, loss_probability):
        emp_dicts.append({"employee_number": i, "loss_probability":j})
    
    emp_dump = json.loads(json.dumps(emp_dicts))
    emp_data = jsonify(emp_dump)
    return emp_data

@app.route("/attrition_by_factor",methods=["GET","POST"])
def tool_data():

    #OVERALL 
    y=df['Employment Status'].value_counts()
    retained=(y[0]/len(df)*100).round(1)
    turnover=(y[1]/len(df)*100).round(1)

    overall_dict = {}
    overall_dict['factor']='Overall'
    overall_dict['Category']=['All']
    overall_dict['Count']=[len(df)]
    overall_dict['Active']= [retained]
    overall_dict['Terminated']=[turnover]
    
    # GENDER
  
    gender_df=df.groupby(["Gender","Employment Status"]).count().reset_index()
    gender_df = gender_df["Age"].astype(float) 

    fem_act_count=gender_df[0]
    fem_term_count=gender_df[1]
    fem_count=fem_act_count+fem_term_count
    male_act_count=gender_df[2]
    male_term_count=gender_df[3]
    male_count=male_act_count+male_term_count
    fem_act_rate=(fem_act_count/fem_count*100).round(1)
    fem_term_rate=(fem_term_count/fem_count*100).round(1)
    male_act_rate=(male_act_count/male_count*100).round(1)
    male_term_rate=(male_term_count/male_count*100).round(1)
    print(male_act_rate,male_term_rate)

    gender_dict={}

    gender_dict['factor'] = 'Gender'
    gender_dict['Category'] = ['Male','Female']
    gender_dict['Count'] = [male_count, fem_count]
    gender_dict['Active'] = [male_act_rate, fem_act_rate]
    gender_dict['Terminated'] = [male_term_rate, fem_term_rate]
    

    # AGE
    count18to29 = 0
    count30to39 = 0
    count40to49 = 0
    count50to59 = 0
    count60up = 0

    for x in df['Age']:
        
        if x >=18 and x<30:
            count18to29 += 1              
        
        elif x >= 30 and x<40:
            count30to39 += 1
                
        elif x >= 40 and x<50:
            count40to49 += 1
            
        elif x >= 50 and x<60:
            count50to59 += 1
        
        elif x >= 60: 
            count60up += 1
            
        else: pass
    

    age18to29_df= df.loc[(df['Age']>=18) & (df['Age']<30)]
    age18to29_act_df=age18to29_df.loc[(df['Employment Status'] == 'Active')]
    age18to29_term_df=age18to29_df.loc[(df['Employment Status'] == 'Terminated')]

    age30to39_df= df.loc[(df['Age']>=30) & (df['Age']<40)]
    age30to39_act_df=age30to39_df.loc[(df['Employment Status'] == 'Active')]
    age30to39_term_df=age30to39_df.loc[(df['Employment Status'] == 'Terminated')]

    age40to49_df= df.loc[(df['Age']>=40) & (df['Age']<50)]
    age40to49_act_df=age40to49_df.loc[(df['Employment Status'] == 'Active')]
    age40to49_term_df=age40to49_df.loc[(df['Employment Status'] == 'Terminated')]

    age50to59_df= df.loc[(df['Age']>=50) & (df['Age']<60)]
    age50to59_act_df=age50to59_df.loc[(df['Employment Status'] == 'Active')]
    age50to59_term_df=age50to59_df.loc[(df['Employment Status'] == 'Terminated')]

    age60up_df= df.loc[(df['Age']>=60)]
    age60up_act_df=age60up_df.loc[(df['Employment Status'] == 'Active')]
    age60up_term_df=age60up_df.loc[(df['Employment Status'] == 'Terminated')]


    per18to29Act=round((len(age18to29_act_df)/count18to29*100),1)
    per18to29Term=round((len(age18to29_term_df)/count18to29*100),1)
    per30to39Act=round((len(age30to39_act_df)/count30to39*100),1)
    per30to39Term=round((len(age30to39_term_df)/count30to39*100),1)
    per40to49Act=round((len(age40to49_act_df)/count40to49*100),1)
    per40to49Term=round((len(age40to49_term_df)/count40to49*100),1)
    per50to59Act=round((len(age50to59_act_df)/count50to59*100),1)
    per50to59Term=round((len(age50to59_term_df)/count50to59*100),1)
    per60upAct=round((len(age60up_act_df)/count60up*100),1)
    per60upTerm=round((len(age60up_term_df)/count60up*100),1)

    age_dict={}

    age_dict['factor'] = 'Age'
    age_dict['Category'] = ['18 to 29','30 to 39', '40 to 49', '50 to 59', 'Over 60']
    age_dict['Count'] = [count18to29, count30to39, count40to49, count50to59, count60up]
    age_dict['Active'] = [per18to29Act, per30to39Act, per40to49Act, per50to59Act, per60upAct]
    age_dict['Terminated'] = [per18to29Term, per30to39Term, per40to49Term, per50to59Term, per60upTerm]
    
    # PERFORMANCE RATING
    perf_df=df.groupby(["Performance Rating","Employment Status"]).count().reset_index()  
    perf_df=perf_df["Age"].astype(float)
    
    count1_total=0
    count2_total=0
    count3_act=perf_df[0]
    count3_term=perf_df[1]
    count4_act=perf_df[2]
    count4_term=perf_df[3]
    count3_total=count3_act + count3_term
    count4_total=count4_act+count4_term

    rate1_act=0
    rate1_term=0
    rate2_act=0
    rate2_term=0
    rate3_act=(count3_act/count3_total*100).round(1)
    rate3_term=(count3_term/count3_total*100).round(1)
    rate4_act=(count4_act/count4_total*100).round(1)
    rate4_term=(count4_term/count4_total*100).round(1)

    perf_dict={}

    perf_dict['factor'] = 'Performance Rating'
    perf_dict['Category'] = ['1','2','3','4']
    perf_dict['Count'] = [count1_total, count2_total, count3_total, count4_total]
    perf_dict['Active'] = [rate1_act, rate2_act, rate3_act, rate4_act]
    perf_dict['Terminated'] = [rate1_term,rate2_term,rate3_term,rate4_term]
    
   
    #JOB INVOLVEMENT 
    
    jobInv=df.groupby(["Job Involvement","Employment Status"]).count().reset_index()    
    JI=jobInv["Age"].astype(float)

    JI1_act=JI[0]
    JI1_term=JI[1]
    JI1_total=(JI1_act + JI1_term)
    JI2_act=JI[2]
    JI2_term=JI[3]
    JI2_total=(JI2_act + JI2_term)
    JI3_act=JI[4]
    JI3_term=JI[5]
    JI3_total=(JI3_act + JI3_term)
    JI4_act=JI[6]
    JI4_term=JI[7]
    JI4_total=(JI4_act + JI4_term)

    rateJI1_act=(JI1_act/JI1_total*100).round(1)
    rateJI1_term=(JI1_term/JI1_total*100).round(1)
    rateJI2_act=(JI2_act/JI2_total*100).round(1)
    rateJI2_term=(JI2_term/JI2_total*100).round(1)
    rateJI3_act=(JI3_act/JI3_total*100).round(1)
    rateJI3_term=(JI3_term/JI3_total*100).round(1)
    rateJI4_act=(JI4_act/JI4_total*100).round(1)
    rateJI4_term=(JI4_term/JI4_total*100).round(1)

    JI_dict={}

    JI_dict['factor'] = 'Job Involvement'
    JI_dict['Category'] = ['1','2','3','4']
    JI_dict['Count'] = [JI1_total, JI2_total, JI3_total, JI4_total]
    JI_dict['Active'] = [rateJI1_act, rateJI2_act, rateJI3_act, rateJI4_act]
    JI_dict['Terminated'] = [rateJI1_term, rateJI2_term, rateJI3_term, rateJI4_term]
    
    # TRAVEL
    
    new_df=df.groupby(["Business Travel","Employment Status"]).count().reset_index()
    trav_df = new_df["Age"].astype(float) 

    none_act=trav_df[2]
    none_term=trav_df[3]
    none_total=none_act + none_term
    rare_act=trav_df[4]
    rare_term=trav_df[5]
    rare_total=rare_act + rare_term
    freq_act=trav_df[0]
    freq_term=trav_df[1]
    freq_total=freq_act+freq_term

    none_act_rate=(none_act/none_total*100).round(1)
    none_term_rate=(none_term/none_total*100).round(1)
    rare_act_rate=(rare_act/rare_total*100).round(1)
    rare_term_rate=(rare_term/rare_total*100).round(1)
    freq_act_rate=(freq_act/freq_total*100).round(1)
    freq_term_rate=(freq_term/freq_total*100).round(1)

    trav_dict = {}

    trav_dict['factor'] = 'Business Travel'
    trav_dict['Category'] = ['None','Rare','Frequent']
    trav_dict['Count'] = [none_total,rare_total,freq_total]
    trav_dict['Active'] = [none_act_rate,rare_act_rate,freq_act_rate]
    trav_dict['Terminated'] = [none_term_rate, rare_term_rate, freq_term_rate]
    
    # DEPARTMENT
   
    new_dep_df=df.groupby(["Department","Employment Status"]).count().reset_index()
    dept_df = new_dep_df["Age"].astype(float)

    hr_act=dept_df[0]
    hr_term=dept_df[1]
    hr_total=hr_act + hr_term
    rd_act=dept_df[2]
    rd_term=dept_df[3]
    rd_total=rd_act + rd_term
    sales_act=dept_df[4]
    sales_term=dept_df[5]
    sales_total=sales_act+sales_term

    hr_act_rate=(hr_act/hr_total*100).round(1)
    hr_term_rate=(hr_term/hr_total*100).round(1)
    rd_act_rate=(rd_act/rd_total*100).round(1)
    rd_term_rate=(rd_term/rd_total*100).round(1)
    sales_act_rate=(sales_act/sales_total*100).round(1)
    sales_term_rate=(sales_term/sales_total*100).round(1)

    dept_dict = {}

    dept_dict['factor'] = 'Department'
    dept_dict['Category'] = ['HR','R&D','Sales']
    dept_dict['Count'] = [hr_total,rd_total,sales_total]
    dept_dict['Active'] = [hr_act_rate,rd_act_rate,sales_act_rate]
    dept_dict['Terminated'] = [hr_term_rate, rd_term_rate, sales_term_rate]

    # COMMUTE
    
    bins = [0, 10, 20, 30]
    labels=['<10 mi','10-19 mi','20-29 mi']
    groups = df.groupby(['Employment Status', pd.cut(df['Commute (Miles)'], bins=bins, labels=labels)])
    cm=groups.size().reset_index().rename(columns={0:"Count"})
    
    cm1_act=int(cm.iloc[0,2])
    cm1_term=int(cm.iloc[3,2])
    cm1_total=int((cm1_act + cm1_term))
    cm2_act=int(cm.iloc[1,2])
    cm2_term=int(cm.iloc[4,2])
    cm2_total=int((cm2_act + cm2_term))
    cm3_act=int(cm.iloc[2,2])
    cm3_term=int(cm.iloc[5,2])
    cm3_total=int((cm3_act + cm3_term))
    
    rate_cm1_act=round((cm1_act/cm1_total*100),1)
    rate_cm1_term=round((cm1_term/cm1_total*100),1)
    rate_cm2_act=round((cm2_act/cm2_total*100),1)
    rate_cm2_term=round((cm2_term/cm2_total*100),1)
    rate_cm3_act=round((cm3_act/cm3_total*100),1)
    rate_cm3_term=round((cm3_term/cm3_total*100),1)
    

    cm_dict={}

    cm_dict['factor'] = 'Commute (Miles)'
    cm_dict['Category'] = labels
    cm_dict['Count'] = [cm1_total, cm2_total, cm3_total]
    cm_dict['Active'] = [rate_cm1_act, rate_cm2_act, rate_cm3_act]
    cm_dict['Terminated'] = [rate_cm1_term, rate_cm2_term, rate_cm3_term]

     #ENVIRONMENT SATISFACTION 
    ES1_count = 0
    ES2_count = 0
    ES3_count = 0
    ES4_count = 0

    for x in df['Environment Satisfaction']:
        
        if x == 1:
            ES1_count += 1           
        
        elif x == 2:
            ES2_count += 1
            
        elif x == 3:
            ES3_count += 1
            
        elif x == 4:
            ES4_count += 1
                
        else: pass

    ES1_act_df=df.loc[(df['Environment Satisfaction']== 1) & (df['Employment Status'] == 'Active')]
    ES1_term_df=df.loc[(df['Environment Satisfaction']== 1) & (df['Employment Status'] == 'Terminated')]
    ES2_act_df=df.loc[(df['Environment Satisfaction']== 2) & (df['Employment Status'] == 'Active')]
    ES2_term_df=df.loc[(df['Environment Satisfaction']== 2) & (df['Employment Status'] == 'Terminated')]
    ES3_act_df=df.loc[(df['Environment Satisfaction']== 3) & (df['Employment Status'] == 'Active')]
    ES3_term_df=df.loc[(df['Environment Satisfaction']== 3) & (df['Employment Status'] == 'Terminated')]
    ES4_act_df=df.loc[(df['Environment Satisfaction']== 4) & (df['Employment Status'] == 'Active')]
    ES4_term_df=df.loc[(df['Environment Satisfaction']== 4) & (df['Employment Status'] == 'Terminated')]

    perES1Act=round((len(ES1_act_df)/ES1_count*100),1)
    perES1Term=round((len(ES1_term_df)/ES1_count*100),1)
    perES2Act=round((len(ES2_act_df)/ES2_count*100),1)
    perES2Term=round((len(ES2_term_df)/ES2_count*100),1)
    perES3Act=round((len(ES3_act_df)/ES3_count*100),1)
    perES3Term=round((len(ES3_term_df)/ES3_count*100),1)
    perES4Act=round((len(ES4_act_df)/ES4_count*100),1)
    perES4Term=round((len(ES4_term_df)/ES4_count*100),1)

    ES_dict={}

    ES_dict['factor'] = 'Environment Satisfaction'
    ES_dict['Category'] = ['1','2','3','4']
    ES_dict['Count'] = [ES1_count, ES2_count, ES3_count, ES4_count]
    ES_dict['Active'] = [perES1Act,perES2Act,perES3Act,perES4Act]
    ES_dict['Terminated'] = [perES1Term,perES2Term,perES3Term,perES4Term]
       

     #JOB LEVEL 
    JL1_count = 0
    JL2_count = 0
    JL3_count = 0
    JL4_count = 0
    JL5_count = 0

    for x in df['Job Level']:
        
        if x == 1:
            JL1_count += 1              
        
        elif x == 2:
            JL2_count += 1
            
        elif x == 3:
            JL3_count += 1
            
        elif x == 4:
            JL4_count += 1

        elif x ==5:
            JL5_count += 1
                
        else: pass

    JL1_act_df=df.loc[(df['Job Level']== 1) & (df['Employment Status'] == 'Active')]
    JL1_term_df=df.loc[(df['Job Level']== 1) & (df['Employment Status'] == 'Terminated')]
    JL2_act_df=df.loc[(df['Job Level']== 2) & (df['Employment Status'] == 'Active')]
    JL2_term_df=df.loc[(df['Job Level']== 2) & (df['Employment Status'] == 'Terminated')]
    JL3_act_df=df.loc[(df['Job Level']== 3) & (df['Employment Status'] == 'Active')]
    JL3_term_df=df.loc[(df['Job Level']== 3) & (df['Employment Status'] == 'Terminated')]
    JL4_act_df=df.loc[(df['Job Level']== 4) & (df['Employment Status'] == 'Active')]
    JL4_term_df=df.loc[(df['Job Level']== 4) & (df['Employment Status'] == 'Terminated')]
    JL5_act_df=df.loc[(df['Job Level']== 5) & (df['Employment Status'] == 'Active')]
    JL5_term_df=df.loc[(df['Job Level']== 5) & (df['Employment Status'] == 'Terminated')]

    perJL1Act=round((len(JL1_act_df)/JL1_count*100),1)
    perJL1Term=round((len(JL1_term_df)/JL1_count*100),1)
    perJL2Act=round((len(JL2_act_df)/JL2_count*100),1)
    perJL2Term=round((len(JL2_term_df)/JL2_count*100),1)
    perJL3Act=round((len(JL3_act_df)/JL3_count*100),1)
    perJL3Term=round((len(JL3_term_df)/JL3_count*100),1)
    perJL4Act=round((len(JL4_act_df)/JL4_count*100),1)
    perJL4Term=round((len(JL4_term_df)/JL4_count*100),1)
    perJL5Act=round((len(JL5_act_df)/JL5_count*100),1)
    perJL5Term=round((len(JL5_term_df)/JL5_count*100),1)

    JL_dict={}

    JL_dict['factor'] = 'Job Level'
    JL_dict['Category'] = ['1','2','3','4', '5']
    JL_dict['Count'] = [JL1_count, JL2_count, JL3_count, JL4_count, JL5_count]
    JL_dict['Active'] = [perJL1Act,perJL2Act,perJL3Act,perJL4Act,perJL5Act]
    JL_dict['Terminated'] = [perJL1Term,perJL2Term,perJL3Term,perJL4Term,perJL5Term]

     #JOB SATISFACTION 
    js_df=df.groupby(["Job Satisfaction","Employment Status"]).count().reset_index() 
    js=js_df["Age"].astype(float)

    actjs1 = js[0]
    terjs1 = js[1]
    alljs1 = js[0] + js[1]
    actjs2 = js[2]
    terjs2 = js[3]
    alljs2 = js[2] + js[3]
    actjs3 = js[4]
    terjs3 = js[5]
    alljs3 = js[4] + js[5]
    actjs4 = js[6]
    terjs4 = js[7]
    alljs4 = js[6] + js[7]

    rate_actjs1 = (actjs1/alljs1*100).round(1)
    rate_actjs2 = (actjs2/alljs2*100).round(1)
    rate_actjs3 = (actjs3/alljs3*100).round(1)
    rate_actjs4 = (actjs4/alljs4*100).round(1)
    rate_termjs1 = (terjs1/alljs1*100).round(1)
    rate_termjs2 = (terjs2/alljs2*100).round(1)
    rate_termjs3 = (terjs3/alljs3*100).round(1)
    rate_termjs4 = (terjs4/alljs4*100).round(1)

    js_dict = {}

    js_dict['factor'] = 'Job Satisfaction'
    js_dict['Category'] = ['1','2','3','4']
    js_dict['Count'] = [alljs1,alljs2,alljs3,alljs4]
    js_dict['Active'] = [rate_actjs1, rate_actjs2, rate_actjs3, rate_actjs4]
    js_dict['Terminated'] = [rate_termjs1, rate_termjs2, rate_termjs3, rate_termjs4]

    # MONTHLY INCOME
    bins = [1000, 3000, 5000, 7000, 20000]
    labels=['<$3000','$3000-4999','$5000-6999','$7000 and up']
    groups = df.groupby(['Employment Status', pd.cut(df['Monthly Income'], bins=bins, labels=labels)])
    mi=groups.size().reset_index().rename(columns={0:"Count"})

    mi1_act=int(mi.iloc[0,2])
    mi1_term=int(mi.iloc[4,2])
    mi1_total=int((mi1_act + mi1_term))
    mi2_act=int(mi.iloc[1,2])
    mi2_term=int(mi.iloc[5,2])
    mi2_total=int((mi2_act + mi2_term))
    mi3_act=int(mi.iloc[2,2])
    mi3_term=int(mi.iloc[6,2])
    mi3_total=int((mi3_act + mi3_term))
    mi4_act=int(mi.iloc[3,2])
    mi4_term=int(mi.iloc[7,2])
    mi4_total=int((mi4_act + mi4_term))

    rate_mi1_act=round((mi1_act/mi1_total*100),1)
    rate_mi1_term=round((mi1_term/mi1_total*100),1)
    rate_mi2_act=round((mi2_act/mi2_total*100),1)
    rate_mi2_term=round((mi2_term/mi2_total*100),1)
    rate_mi3_act=round((mi3_act/mi3_total*100),1)
    rate_mi3_term=round((mi3_term/mi3_total*100),1)
    rate_mi4_act=round((mi4_act/mi4_total*100),1)
    rate_mi4_term=round((mi4_term/mi4_total*100),1)

    mi_dict={}

    mi_dict['factor'] = 'Monthly Income'
    mi_dict['Category'] = labels
    mi_dict['Count'] = [mi1_total, mi2_total, mi3_total, mi4_total]
    mi_dict['Active'] = [rate_mi1_act, rate_mi2_act, rate_mi3_act, rate_mi4_act]
    mi_dict['Terminated'] = [rate_mi1_term, rate_mi2_term, rate_mi3_term, rate_mi4_term]

    # STOCK OPTIONS
    so_df=df.groupby(["Stock Option Level","Employment Status"]).count().reset_index() 
    so=so_df["Age"].astype(float)

    actso1 = so[0]
    terso1 = so[1]
    allso1 = so[0] + so[1]
    actso2 = so[2]
    terso2 = so[3]
    allso2 = so[2] + so[3]
    actso3 = so[4]
    terso3 = so[5]
    allso3 = so[4] + so[5]
    actso4 = so[6]
    terso4 = so[7]
    allso4 = so[6] + so[7]

    rate_actso1 = (actso1/allso1*100).round(1)
    rate_actso2 = (actso2/allso2*100).round(1)
    rate_actso3 = (actso3/allso3*100).round(1)
    rate_actso4 = (actso4/allso4*100).round(1)
    rate_termso1 = (terso1/allso1*100).round(1)
    rate_termso2 = (terso2/allso2*100).round(1)
    rate_termso3 = (terso3/allso3*100).round(1)
    rate_termso4 = (terso4/allso4*100).round(1)

    so_dict = {}

    so_dict['factor'] = 'Stock Option Level'
    so_dict['Category'] = ['1','2','3','4']
    so_dict['Count'] = [allso1,allso2,allso3,allso4]
    so_dict['Active'] = [rate_actso1, rate_actso2, rate_actso3, rate_actso4]
    so_dict['Terminated'] = [rate_termso1, rate_termso2, rate_termso3, rate_termso4]

    # TRAINING
    bins = [0,1,3,5,7]
    labels=['None','1 or 2', '3 or 4', '5 or 6']

    groupsT = df.groupby(['Employment Status', pd.cut(df['Training Last Year'], bins=bins, labels=labels)])
    tr=groupsT.size().reset_index().rename(columns={0:"Count","Training Last Year":"Trainings Last Year"})

    tr1_act=int(tr.iloc[0,2])
    tr1_term=int(tr.iloc[4,2])
    tr1_total=int((tr1_act + tr1_term))
    tr2_act=int(tr.iloc[1,2])
    tr2_term=int(tr.iloc[5,2])
    tr2_total=int((tr2_act + tr2_term))
    tr3_act=int(tr.iloc[2,2])
    tr3_term=int(tr.iloc[6,2])
    tr3_total=int((tr3_act + tr3_term))
    tr4_act=int(tr.iloc[3,2])
    tr4_term=int(tr.iloc[7,2])
    tr4_total=int((tr4_act + tr4_term))

    rate_tr1_act=round((tr1_act/tr1_total*100),1)
    rate_tr1_term=round((tr1_term/tr1_total*100),1)
    rate_tr2_act=round((tr2_act/tr2_total*100),1)
    rate_tr2_term=round((tr2_term/tr2_total*100),1)
    rate_tr3_act=round((tr3_act/tr3_total*100),1)
    rate_tr3_term=round((tr3_term/tr3_total*100),1)
    rate_tr4_act=round((tr4_act/tr4_total*100),1)
    rate_tr4_term=round((tr4_term/tr4_total*100),1)

    tr_dict={}


    tr_dict['factor'] = 'Trainings Last Year'
    tr_dict['Category'] = labels
    tr_dict['Count'] = [tr1_total, tr2_total, tr3_total, tr4_total]
    tr_dict['Active'] = [rate_tr1_act, rate_tr2_act, rate_tr3_act, rate_tr4_act]
    tr_dict['Terminated'] = [rate_tr1_term, rate_tr2_term, rate_tr3_term, rate_tr4_term]

    # CREATE JSON OBJECT
    combined_dicts = [overall_dict, age_dict, trav_dict, cm_dict, dept_dict, ES_dict, gender_dict, JI_dict, JL_dict, js_dict, 
    mi_dict, perf_dict, so_dict, tr_dict]
    tool_data = json.loads(json.dumps(combined_dicts))
     
    return jsonify(tool_data)    

if __name__ == "__main__":
    app.run(debug=True)