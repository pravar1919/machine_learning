import numpy as np
import math
import pandas as pd
from sklearn import linear_model


def from_modal(df):
    reg = linear_model.LinearRegression()
    reg.fit(df[['math']], df.cs)
    return reg.coef_, reg.intercept_


def gradient_descent(x, y):
    m_curr = b_curr = 0
    iteration = 1000000
    n = len(x)
    learning_rate = 0.0002
    cost_old = 0
    for i in range(iteration):
        y_pred = m_curr*x + b_curr
        cost = (1/n)*sum([i**2 for i in (y-y_pred)])
        md = -(2/n)*sum(x*(y-y_pred))
        bd = -(2/n)*sum((y-y_pred))
        m_curr = m_curr - (learning_rate * md)
        b_curr = b_curr - (learning_rate * bd)
        if math.isclose(cost, cost_old, rel_tol=1e-20):
            break
        cost_old = cost
        # print(f"m: {m_curr}; b: {b_curr}; cost: {cost}, {i} iterations")
    return m_curr, b_curr


if __name__ == "__main__":
    df = pd.read_csv("test_scores.csv")
    x = np.array(df.math)
    y = np.array(df.cs)
    m_curr, b_curr = gradient_descent(x, y)
    print(f"{m_curr}, {b_curr} from algo.")

    m_curr, b_curr = from_modal(df)
    print(f"{m_curr}, {b_curr} from Modal.")
