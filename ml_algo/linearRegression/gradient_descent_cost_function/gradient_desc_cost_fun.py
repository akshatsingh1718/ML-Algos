import numpy as np
import pickle

def gradient_descent(x, y):
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.08
    for i in range(iterations):
        yp = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-yp)])
        md = -(2/n) * sum(x*(y-yp))
        bd = -(2/n) * sum(y-yp)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("m:{}, b:{}, cost:{}, iterations:{}".format(m_curr, b_curr, cost, i))


if __name__ == '__main__':
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 7, 9, 11, 13])

    gradient_descent(x, y)


