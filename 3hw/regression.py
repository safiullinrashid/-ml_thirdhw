import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Загрузка данных
bike_data = pd.read_csv('bikes_rent.csv')

# Задача 2: Простая линейная регрессия
weather_condition = bike_data[['weathersit']].values
demand = bike_data['cnt'].values

linear_model = LinearRegression()
linear_model.fit(weather_condition, demand)

plt.scatter(weather_condition, demand, color='green')
plt.plot(weather_condition, linear_model.predict(weather_condition), color='red')
plt.title('Прогноз спроса на основе погодных условий')
plt.xlabel('Погодные условия')
plt.ylabel('Спрос')
plt.show()

# Задача 3: Предсказание значения cnt
new_weather_condition = np.array([[4]])
predicted_demand = linear_model.predict(new_weather_condition)
print(f'Предсказанное количество аренд: {predicted_demand[0]}')

# Задача 4: Уменьшение размерности и построение 2D графика
reducer = PCA(n_components=2)
reduced_data = reducer.fit_transform(bike_data.drop('cnt', axis=1))

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=demand)
plt.title('2D график предсказания cnt')
plt.show()

# Задача 5: Регуляризация Lasso
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(bike_data.drop('cnt', axis=1), demand)

# Определение признака, оказывающего наибольшее влияние на cnt
feature_coefficients = pd.Series(lasso_model.coef_, index=bike_data.drop('cnt', axis=1).columns)
print(f'Признак, оказывающий наибольшее влияние на cnt: {feature_coefficients.idxmax()}')