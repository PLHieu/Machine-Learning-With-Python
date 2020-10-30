import pandas as pd 
import numpy as np 
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# load data from sklearn forder
from sklearn.datasets import load_boston
boston = load_boston()

# split into x y
df_x = pd.DataFrame(boston.data, columns = boston.feature_names)
df_y = pd.DataFrame(boston.target)

# slit trainning and testing
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.33, random_state = 42)

# init model
reg = linear_model.LinearRegression()

# train model 
reg.fit(x_train, y_train)

# print the coeffecients
# print(reg.coef_)

y_test_numpy  = y_test.values;
#print the predition on our test datas
y_predict = reg.predict(x_test)

for i in range(len(y_predict)):
    print(y_predict[i], y_test_numpy[i])



