import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


df=pd.read_csv('diabetes.csv')

df['Glucose']=np.where(df['Glucose']==0,df['Glucose'].mean(),df['Glucose'])
df['Insulin']=np.where(df['Insulin']==0,df['Insulin'].mean(),df['Insulin'])

df['Pregnancies']=np.where(df['Pregnancies']==0,df['Pregnancies'].mean(),df['Pregnancies'])
df['SkinThickness']=np.where(df['SkinThickness']==0,df['SkinThickness'].mean(),df['SkinThickness'])
# print(df)
# print(df['Insulin'])
X=df.drop('Outcome',axis=1)
y=df['Outcome']

from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=33,test_size=0.2)


# rf=RandomForestClassifier(n_estimators=10)
# rf.fit(Xtrain,ytrain)
# prediction=rf.predict(Xtest)
# from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
# print(classification_report(ytest,prediction))
# print(confusion_matrix(ytest,prediction))
# print(accuracy_score(ytest,prediction))

# # manual hyperparametetuning 
# model=RandomForestClassifier(n_estimators=300,criterion='entropy',min_samples_leaf=10,random_state=100,max_features='sqrt')
# model.fit(Xtrain,ytrain)
# prediction=model.predict(Xtest)
# from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
# print(classification_report(ytest,prediction))
# print(confusion_matrix(ytest,prediction))
# print(accuracy_score(ytest,prediction))

# from sklearn.model_selection import GridSearchCV,KFold,RandomizedSearchCV
# rf=RandomForestClassifier()
# rv=RandomizedSearchCV()
# criteration=['entropy,gini']
# n_estimators=[int(x) for x in np.linspace(start=200,stop=2000,num=10)]
# max_features=['auto','sqrt','log2',]
# max_depth=[int(x) for x in np.linspace(10,1000,10)]
# min_sample=[1,3,5,7,9,11]
# min_sample_leaf=[1,2,4,6,8]
# randomgrid={'n_estimators': n_estimators,
#             'max_features':max_features,
#             'max_depth':max_depth,
#             'min_sample':min_sample,
#             'min_sample_leaf':min_sample_leaf,
#             'criteration':criteration
# }
# rv=RandomizedSearchCV(estimator=rf,param_distributions=randomgrid,n_iter=100,cv=3,verbose=2,random_state=100,n_jobs=-1)
# rv.fit(Xtrain,ytrain)
# print(rv.best_params_)
# best_random_rid=rv.best_estimator_

# automated hyperparameter tuning
# bayesian optimization 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
# model=RandomForestClassifier()
# model.fit(Xtrain,ytrain)
# ypred=model.predict(Xtest)
from sklearn.metrics import f1_score,fbeta_score,roc_auc_score,auc,precision_score,mean_absolute_error,root_mean_squared_error,r2_score,classification_report,accuracy_score
from sklearn.model_selection import StratifiedKFold,cross_val_score
# ✅ Define the search space correctly
# space = {
#     "criterion": hp.choice('criterion', ['entropy', 'gini']),
#     "max_depth": hp.quniform("max_depth", 10, 1200, 10),
#     "max_features": hp.choice("max_features", ['sqrt',  'log2', None]),
#     "min_samples_leaf": hp.uniform("min_samples_leaf", 0.0, 0.5),
#     "min_samples_split": hp.uniform("min_samples_split", 0.0, 1.0),
#     "n_estimators": hp.choice('n_estimators', [10, 50, 300, 750, 1200])
# }

# # ✅ Objective function
# def objective(space):
#     model = RandomForestClassifier(
#         criterion=space['criterion'],
#         max_depth=int(space['max_depth']),
#         max_features=space['max_features'],
#         min_samples_leaf=space['min_samples_leaf'],
#         min_samples_split=space['min_samples_split'],
#         n_estimators=space['n_estimators']
#     )

#     accuracy = cross_val_score(model, Xtrain, ytrain, cv=5).mean()
#     return {'loss': -accuracy, 'status': STATUS_OK}


# trials = Trials()
# best = fmin(
#     fn=objective,
#     space=space,
#     algo=tpe.suggest,
#     max_evals=80,
#     trials=trials
# )
rf=RandomForestClassifier(criterion='gini',max_features=None,min_samples_leaf=0.5265,min_samples_split=0.11,n_estimators=300,max_depth=650)
kfold=StratifiedKFold(n_splits=5)
print(cross_val_score(rf,X,y,cv=kfold))
rf.fit(Xtrain,ytrain)
ypred=rf.predict(Xtest)
# print(classification_report(ytest,ypred))
# print(roc_auc_score(ytest,ypred))
# print(accuracy_score(ytest,ypred))
# print(fbeta_score(ytest,ypred,beta=2,average='weighted'))

# print(best)
# from sklearn.neighbors import KNeighborsClassifier
# kf=KNeighborsClassifier()
pregnancy=float(input("enter preganency index ranges from 0 t0 10"))
glucose=float(input("enter your glucose"))
bp=int(input("enter your bp"))
skinthickness=float(input("enter your skinthickness"))
insulin=float(input("enter your insulin"))
BMI=float(input("enter your bmi"))
diabetesfunction=float(input("diabetespedigree function"))
Age=int(input("enter your age"))
userinput=np.array([[pregnancy,glucose,bp,skinthickness,insulin,BMI,diabetesfunction,Age]])
prediction=rf.predict(userinput)
if prediction==1:
    print("yes you have diabetes")
else:
    print("You dont have diabetes")

df.Age.hist()
plt.show()