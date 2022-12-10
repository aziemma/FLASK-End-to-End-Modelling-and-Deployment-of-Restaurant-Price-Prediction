import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from  sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('Zomato_df.csv')
#print(df.head())

df.drop('Unnamed: 0', axis=1, inplace=True)

x=df.drop('rate',axis=1)
y=df['rate']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

et_model=ExtraTreesRegressor()
et_model.fit(x_train,y_train)
y_pred=et_model.predict(x_test)

import pickle
pickle.dump(et_model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl', 'rb'))
print(y_pred)


