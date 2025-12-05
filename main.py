import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


df=pd.read_csv('diabetes.csv')

df['Glucose']=np.where(df['Glucose']==0,df['Glucose'].mean(),df['Glucose'])
df['Insulin']=np.where(df['Insulin']==0,df['Insulin'].mean(),df['Insulin'])

df['Pregnancies']=np.where(df['Pregnancies']==0,df['Pregnancies'].mean(),df['Pregnancies'])
df['SkinThickness']=np.where(df['SkinThickness']==0,df['SkinThickness'].mean(),df['SkinThickness'])
# print(df)
# print(df['Insulin'])
X=df.drop(['Outcome','Pregnancies','DiabetesPedigreeFunction','SkinThickness','BloodPressure'],axis=1)
y=df['Outcome']

from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=33,test_size=0.2)


rf=RandomForestClassifier(n_estimators=10)
rf.fit(Xtrain,ytrain)
prediction=rf.predict(Xtest)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(ytest,prediction))
print(confusion_matrix(ytest,prediction))
print(accuracy_score(ytest,prediction))

# manual hyperparametetuning 
model=RandomForestClassifier(n_estimators=300,criterion='entropy',min_samples_leaf=10,random_state=100,max_features='sqrt')
model.fit(Xtrain,ytrain)
prediction=model.predict(Xtest)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(ytest,prediction))
print(confusion_matrix(ytest,prediction))
print(accuracy_score(ytest,prediction))

from sklearn.model_selection import GridSearchCV,KFold,RandomizedSearchCV
rf=RandomForestClassifier()
rv=RandomizedSearchCV()
criteration=['entropy,gini']
n_estimators=[int(x) for x in np.linspace(start=200,stop=2000,num=10)]
max_features=['auto','sqrt','log2',]
max_depth=[int(x) for x in np.linspace(10,1000,10)]
min_sample=[1,3,5,7,9,11]
min_sample_leaf=[1,2,4,6,8]
randomgrid={'n_estimators': n_estimators,
            'max_features':max_features,
            'max_depth':max_depth,
            'min_sample':min_sample,
            'min_sample_leaf':min_sample_leaf,
            'criteration':criteration
}
rv=RandomizedSearchCV(estimator=rf,param_distributions=randomgrid,n_iter=100,cv=3,verbose=2,random_state=100,n_jobs=-1)
rv.fit(Xtrain,ytrain)
print(rv.best_params_)
best_random_rid=rv.best_estimator_

automated hyperparameter tuning
bayesian optimization 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.metrics import f1_score,fbeta_score,roc_auc_score,auc,precision_score,mean_absolute_error,root_mean_squared_error,r2_score,classification_report,accuracy_score
from sklearn.model_selection import StratifiedKFold,cross_val_score

space = {
    "criterion": hp.choice('criterion', ['entropy', 'gini']),
    "max_depth": hp.quniform("max_depth", 10, 1200, 10),
    "max_features": hp.choice("max_features", ['sqrt',  'log2', None]),
    "min_samples_leaf": hp.uniform("min_samples_leaf", 0.0, 0.5),
    "min_samples_split": hp.uniform("min_samples_split", 0.0, 1.0),
    "n_estimators": hp.choice('n_estimators', [10, 50, 300, 750, 1200])
}


def objective(space):
    model = RandomForestClassifier(
        criterion=space['criterion'],
        max_depth=int(space['max_depth']),
        max_features=space['max_features'],
        min_samples_leaf=space['min_samples_leaf'],
        min_samples_split=space['min_samples_split'],
        n_estimators=space['n_estimators']
    )

    accuracy = cross_val_score(model, Xtrain, ytrain, cv=5).mean()
    return {'loss': -accuracy, 'status': STATUS_OK}


trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=80,
    trials=trials
)


rf=RandomForestClassifier(criterion='gini',max_features=None,min_samples_leaf=0.5265,min_samples_split=0.11,n_estimators=300,max_depth=650)
kfold=StratifiedKFold(n_splits=5)
print(cross_val_score(rf,X,y,cv=kfold))
rf.fit(Xtrain,ytrain)
ypred=rf.predict(Xtest)

print(classification_report(ytest,ypred))
print(roc_auc_score(ytest,ypred))
print(accuracy_score(ytest,ypred))
print(fbeta_score(ytest,ypred,beta=2,average='weighted'))

print(best)

'''feature selection'''
# 1st method
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
order_rank=SelectKBest(score_func=chi2,k=8)

variabel=order_rank.fit(X,y)

a=pd.DataFrame(variabel.scores_,columns=['Score'])
b=pd.DataFrame(X.columns)

c=pd.concat([a,b],axis=1)
print(c.nlargest(8,'Score'))
'''2nd method'''
from sklearn.ensemble import ExtraTreesClassifier
ex=ExtraTreesClassifier()
ex.fit(X,y)
rank=pd.Series(ex.feature_importances_,index=X.columns)

print(rank.nlargest(8))
information gain
from sklearn.feature_selection import mutual_info_classif
mutual=mutual_info_classif(X,y)
mutual_data=pd.Series(mutual,index=X.columns)
print(mutual_data.sort_values(ascending=False))

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

rfe=RFE(estimator=RandomForestClassifier(),n_features_to_select=4)
rfe.fit(X,y)
for i, col in enumerate(X.columns):
    print(f"{col} selected={rfe.support_[i]} rank={rfe.ranking_[i]}")
    





"""from all the methods we can say that the most important feature for diabetes prediction is 
1 glucose
2 insulin
3 Bmi
4 Age
"""
first lets balance the data set
from imblearn.combine import SMOTETomek
sm=SMOTETomek(sampling_strategy='auto')
Xtrain_ns,ytrain_ns=sm.fit_resample(Xtrain,ytrain)


# now lets make a correlation heat map 

sns.set_context("notebook")
sns.set_style("darkgrid")
sns.set_palette("colorblind")
corr=df[['Age','Glucose','Insulin','BMI']].corr()

sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.show()


# now lets make a little advanced plot for the entities 


fig=px.scatter(df,x='Age',y='Glucose',color='BMI',animation_frame='Outcome')
fig=px.scatter(df,x='Age',y='Insulin',color='BMI',animation_frame='Outcome')
fig.show()

rf=RandomForestClassifier(criterion='gini',max_features=None,min_samples_leaf=0.5265,min_samples_split=0.11,n_estimators=300,max_depth=650)
kfold=StratifiedKFold(n_splits=5)
rf.fit(Xtrain_ns,ytrain_ns)

user_Age=int(input("Enter your AGE: "))
user_insulin=float(input("Enter your insulin: "))
user_BMI=float(input("Enter your BMI: "))
user_GLUCOSE=int(input("Enter your GLUCOSE: "))
user_Input=np.array([[user_Age,user_insulin,user_BMI,user_GLUCOSE,]])
prediction=rf.predict(user_Input)
if prediction==1:
    print('You have diabetes')
else:
    print("You dont have diabetes")
    

