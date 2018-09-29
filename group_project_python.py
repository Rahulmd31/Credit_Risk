
# coding: utf-8

# In[88]:


#Importing libraries.
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
import datetime
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import cross_validation
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import warnings


# In[89]:


#Load data into pandas.
Data = pd.read_csv(r'C:\Users\00008020\.spyder-py3\python_programs\python_codes\XYZCorp_LendingData.txt', delimiter='\t',low_memory=False,header =0)

loan_data = pd.DataFrame(Data)


# In[90]:


# count half point of the dataset.Remove NA values more than 50%.
half_point = len(loan_data) / 2
print (half_point)
loan_data = loan_data.dropna(thresh=half_point, axis=1)


# In[33]:


loan_data.isnull().sum()
#(855969, 52)


# In[92]:


#eliminating irrelevant values
#application type have almost constant value:individual and  442 obs have value:joint
#constant values doesn't affect regression
#address state have too many unique values
drop_list = ['id','member_id','sub_grade','emp_title','zip_code','addr_state','pymnt_plan',
             'application_type','next_pymnt_d','last_credit_pull_d','last_pymnt_d','earliest_cr_line',
             'initial_list_status']
loan_data = loan_data.drop(drop_list,axis=1)

#using date variable type 1 error is increasing thats why we are eliminating them and also
#date variable requires potential engineering thats why its better to drop all the date variables
#initial list status:type 1 error increases if this feature is included,hence we eliminated it
#(855969, 39)


# In[51]:


for column in loan_data.columns:
    if (len(loan_data[column].unique()) < 4):
        print(loan_data[column].value_counts())
        print()


# In[94]:


loan_data=loan_data.drop("policy_code",axis=1)
#(855969, 38)


# In[40]:


# purpose & title columns contains overlapping information. But purpose column contains fewer descrete values & is cleaner 

for name in ['purpose','title']:
    print("Unique Values in column: {}\n".format(name)) 
    print(loan_data[name].value_counts(),'\n')


# In[96]:


# we drop title here based on above obs
loan_data = loan_data.drop("title", axis=1)
#(855969, 37)


# In[98]:


# Filling NA values
loan_data['emp_length']=loan_data['emp_length'].fillna(0)
col_name2=['revol_util','collections_12_mths_ex_med',
'tot_coll_amt','tot_cur_bal','total_rev_hi_lim']
for x in col_name2: loan_data[x].fillna(loan_data[x].mean(),inplace=True)


# In[99]:


#we are getting mode value as 10+year in emp_length so it can't be consider for filling NA.here all the null
#obs are filled with '0' in above code block and for rest mapping is used below.
mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
    }
}
loan_data= loan_data.replace(mapping_dict)
loan_data['emp_length'].head()


# In[100]:


#%%Label encoding
colname=["grade","home_ownership", "verification_status", "purpose",
         "term"]
le={}
for x in colname:
    le[x]=preprocessing.LabelEncoder()

for x in colname:
    loan_data[x]=le[x].fit_transform(loan_data.__getattr__(x))


# In[84]:


#Multicolinearity
cor =loan_data.corr() 
cor.loc[:,:] = np.tril(cor, k=-2) # below main lower triangle of an array

cor = cor.stack()
cor[(cor > 0.9) | (cor < -0.9)]#only highly correlated values are considered here


# In[103]:


plt.figure(figsize=(20,14))
sns.heatmap(loan_data.corr(), annot=True, linewidths=.5,fmt='.2f')
plt.title('Correlation Heat Map')


# In[104]:


#based on multicollinearity we are droping the below mentioned columns
loan_data=loan_data.drop(['int_rate',"funded_amnt_inv","installment","total_rec_prncp",],axis=1)
#(855969, 33)


# In[105]:


loan_data.isnull().sum()


# In[107]:


print("We've been able to reduced the features to => {}".format(loan_data.shape))


# In[108]:


#%%Splitting data using'issue_d'.

loan_data['issue_d']=pd.to_datetime(loan_data['issue_d'],format='%b-%Y')

train_data=loan_data[loan_data['issue_d']<'2015-06-01']

test_data=loan_data[loan_data['issue_d'] >='2015-06-01']

train_data = train_data.drop("issue_d",axis=1)
test_data = test_data.drop("issue_d",axis=1)


x_train=train_data.values[:,:-1]#all columns except last,negative indexing
y_train=train_data.values[:,-1]#all rows of last column
x_test=test_data.values[:,:-1]
y_test=test_data.values[:,-1] 


# In[109]:


#%%------------------Logistic----------------------------------

#create model
classifier=(LogisticRegression())
#fitting training data to model
classifier.fit(x_train,y_train)
#predicting
y_pred=classifier.predict(x_test)


# In[110]:


#creating confusion matrix

cfm=confusion_matrix(y_test,y_pred)
print(cfm)
print("Classification report:")
print(classification_report(y_test,y_pred))
acc=accuracy_score(y_test,y_pred)
print("accuracy of model is:",acc)


# In[111]:


y_pred_prob=classifier.predict_proba(x_test)
print(y_pred_prob)


# In[112]:


#Adusting threshod.
for a in np.arange(0,1,0.05):
    predict_mine=np.where(y_pred_prob[:,0]<a,1,0)
    cfm=confusion_matrix(y_test.tolist(),predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("errors at threshold",a,":",total_err,"type 2 error :",          cfm[1,0],"type 1 error:",cfm[0,1])


# In[113]:


y_pred_class=[]
for value in y_pred_prob[:,0]:
    if value <0.4:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)


# In[114]:


cfm=confusion_matrix(y_test.tolist(),y_pred_class)
print(cfm)
acc=accuracy_score(y_test.tolist(),y_pred_class)
print("accuracy of model is:",acc) 
print("Classification report:") 
print(classification_report(y_test.tolist(),y_pred_class))  


# In[115]:


fpr,tpr,threshold=metrics.roc_curve(y_test.tolist(),y_pred_class)
auc=metrics.auc(fpr,tpr)
print("auc:",auc)


# In[116]:


plt.title("receiver operating characteristics")
plt.plot(fpr,tpr,'r',label=auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'g--')
plt.xlim([0,1])   
plt.ylim([0,1])
plt.xlabel('false +ve rate')
plt.ylabel('true +ve rate') 
#plt.show()


# In[117]:


#k-folds cross validation used for evaluation of model
#in k-folds cross-validation no drastic change in accuracy score is found hence original model is preferred
classifier=(LogisticRegression())
from sklearn import cross_validation
#performing kfold_cross_validation
kfold_cv=cross_validation.KFold(n=len(x_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=x_train,
y=y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())

for train_value, test_value in kfold_cv:
    classifier.fit(x_train[train_value], y_train[train_value]).predict(x_train[test_value])


    y_pred=classifier.predict(x_test)
warnings.filterwarnings('ignore')


# In[118]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(y_test,y_pred)
print(cfm)
acc=accuracy_score(y_test,y_pred)
print("accuracy of model is:",acc) 
print("Classification report:") 
print(classification_report(y_test,y_pred)) 


# In[70]:


from sklearn.ensemble import AdaBoostClassifier
classifier=(LogisticRegression())
model_AdaBoost=(AdaBoostClassifier(base_estimator=classifier))
model_AdaBoost.fit(x_train,y_train)
y_pred=model_AdaBoost.predict(x_test)
warnings.filterwarnings('ignore')


# In[ ]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
#[[253872   2703]
# [    60    251]]
#0.9892442562070335
#            precision    recall  f1-score   support

#      0.0       1.00      0.99      0.99    256575
#     1.0       0.08      0.81      0.15       311

#avg / total       1.00      0.99      0.99    256886'''


# In[122]:


test_data["predicted_y"]=y_pred_class

test_data.to_csv(r"C:\Users\00008020\.spyder-py3\python_programs\python_codes\test_data_predicted.csv",index=False)

