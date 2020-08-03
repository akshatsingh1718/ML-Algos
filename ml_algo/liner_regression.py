import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# dict_keys(['data', 'target', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
diabetes = datasets.load_diabetes()
# diabetes_x = diabetes.data[:, np.newaxis, 2]
diabetes_x = diabetes.data

diabetes_x_train = diabetes_x[:-30]
diabetes_x_test = diabetes_x[-30:]

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()

model.fit(diabetes_x_train, diabetes_y_train)
diabetes_y_predicted = model.predict(diabetes_x_test)
print("Mean error is :", mean_squared_error(diabetes_y_test, diabetes_y_predicted))
print("Weights (tan{theta}):", model.coef_)
print("Intercepts (w{not}):", model.intercept_)

# plt.scatter(diabetes_x_test, diabetes_y_test)
# plt.plot(diabetes_x_test, diabetes_y_predicted)
# plt.show()

''' Mean with only one weight
Mean error is : 3035.0601152912695
Weights (tan{theta}): [941.43097333]
Intercepts : 153.39713623331698
'''
