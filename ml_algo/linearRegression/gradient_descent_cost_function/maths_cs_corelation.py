import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math


def predict_using_sklearn():
    df = pd.read_csv('students.csv')
    reg = LinearRegression()
    reg.fit(df[['math']], df.cs)
    return reg.coef_, reg.intercept_


def gradient_descent(x, y):
    b_curr = m_curr = 0
    n = len(x)
    iterations = 1000000
    learning_rate = 0.0002
    cost_previous = 0
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)])
        md = -(2/n) * sum(x * (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost
        print('cost:{}, m:{}, b:{}, iteration:{}'.format(cost, m_curr, b_curr, i))
    print("y : ", y_predicted)
    return m_curr, b_curr


if __name__ == '__main_':
    student_df = pd.read_csv('students.csv')
    maths = np.array(student_df.math)
    cs = np.array(student_df.cs)

    m, b = gradient_descent(maths, cs)
    print('Using gradient descent function: Coef {} Intercept:{}'.format(m, b))

    m_sklearn, b_sklearn = predict_using_sklearn()
    print('Using gradient descent function: Coef {} Intercept:{}'.format(m_sklearn, b_sklearn))

