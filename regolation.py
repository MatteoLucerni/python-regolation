import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

boston = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", sep=r'\s+', names=["CRIM", "ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PRATIO","B","LSTAT","MEDV"])
boston.head()

# simulazione un caso di overfitting

X = boston.drop('MEDV', axis=1).values
Y = boston['MEDV'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

pf = PolynomialFeatures(degree=2)

X_train_poly = pf.fit_transform(X_train)
X_test_poly = pf.transform(X_test)

ss = StandardScaler()

X_train_poly = ss.fit_transform(X_train_poly)
X_test_poly = ss.transform(X_test_poly)

lr = LinearRegression()
lr.fit(X_train_poly, Y_train)

# test
Y_pred = lr.predict(X_test_poly)

mse = mean_squared_error(Y_test, Y_pred)
r2s = r2_score(Y_test, Y_pred)

print('Test Score: ' +  str(r2s) + ' / Error: ' + str(mse))

# train
Y_pred_train = lr.predict(X_train_poly)

mse_train = mean_squared_error(Y_train, Y_pred_train)
r2s_train = r2_score(Y_train, Y_pred_train)

print('Train Score: ' +  str(r2s_train) + ' / Error: ' + str(mse_train))

# risoluzione overfitting con regolazione L2

from sklearn.linear_model import Ridge

print('---------- Regolation L2 Ridge ------------------')
print('---------- Regolation L2 Ridge ------------------')
print('---------- Regolation L2 Ridge ------------------')

alphas = [0.0001, 0.001, 0.01, 0.1, 1., 10.]
for alpha in alphas:
    print('ALPHA: ' + str(alpha))
    model = Ridge(alpha=alpha)
    model.fit(X_train_poly, Y_train)

    Y_pred_train = model.predict(X_train_poly)
    Y_pred_test = model.predict(X_test_poly)

    mse_train = mean_squared_error(Y_train, Y_pred_train)
    r2s_train = r2_score(Y_train, Y_pred_train)

    mse_test = mean_squared_error(Y_test, Y_pred_test)
    r2s_test = r2_score(Y_test, Y_pred_test)

    print('Test Score: ' +  str(r2s_test) + ' / Error: ' + str(mse_test))
    print('Train Score: ' +  str(r2s_train) + ' / Error: ' + str(mse_train))

# risoluzione overfitting con regolazione L1

from sklearn.linear_model import Lasso

print('---------- Regolation L1 Lasso ------------------')
print('---------- Regolation L1 Lasso ------------------')
print('---------- Regolation L1 Lasso ------------------')

alphas = [0.0001, 0.001, 0.01, 0.1, 1., 10.]
for alpha in alphas:
    print('ALPHA: ' + str(alpha))
    model = Lasso(alpha=alpha)
    model.fit(X_train_poly, Y_train)

    Y_pred_train = model.predict(X_train_poly)
    Y_pred_test = model.predict(X_test_poly)

    mse_train = mean_squared_error(Y_train, Y_pred_train)
    r2s_train = r2_score(Y_train, Y_pred_train)

    mse_test = mean_squared_error(Y_test, Y_pred_test)
    r2s_test = r2_score(Y_test, Y_pred_test)

    print('Test Score: ' +  str(r2s_test) + ' / Error: ' + str(mse_test))
    print('Train Score: ' +  str(r2s_train) + ' / Error: ' + str(mse_train))

from sklearn.linear_model import ElasticNet

print('---------- Regolation L1 & L2 ElasticNet ------------------')
print('---------- Regolation L1 & L2 ElasticNet ------------------')
print('---------- Regolation L1 & L2 ElasticNet ------------------')

alphas = [0.0001, 0.001, 0.01, 0.1, 1., 10.]
for alpha in alphas:
    print('ALPHA: ' + str(alpha))
    # l1_ratio per decidere su quale delle due regolarizzazioni sbilanciarsi
    model = ElasticNet(alpha=alpha, l1_ratio=0.5)
    model.fit(X_train_poly, Y_train)

    Y_pred_train = model.predict(X_train_poly)
    Y_pred_test = model.predict(X_test_poly)

    mse_train = mean_squared_error(Y_train, Y_pred_train)
    r2s_train = r2_score(Y_train, Y_pred_train)

    mse_test = mean_squared_error(Y_test, Y_pred_test)
    r2s_test = r2_score(Y_test, Y_pred_test)

    print('Test Score: ' +  str(r2s_test) + ' / Error: ' + str(mse_test))
    print('Train Score: ' +  str(r2s_train) + ' / Error: ' + str(mse_train))

