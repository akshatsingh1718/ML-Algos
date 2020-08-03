import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("homeprices.csv")
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

# print(reg.predict([[3300]]))

# print("Coeff(m) : ", reg.coef_)
# print("intercept(c) :", reg.intercept_)

# predicting areas from areas.csv and saving data in predicted_areas.csv
d = pd.read_csv("areas.csv")
areas_prices_predictions = reg.predict(d)
d['prices'] = areas_prices_predictions
d.to_csv('predicted_areas.csv', index=False)

# print(3300*135.78767123+180616.43835616432)
plt.xlabel("area")
plt.ylabel("price")
plt.scatter(df.area, df.price, color="red", marker="+")
plt.plot(d.area, areas_prices_predictions)
plt.show()

