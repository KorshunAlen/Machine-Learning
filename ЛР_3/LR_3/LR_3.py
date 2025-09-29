# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Загрузка данных и подготовка
df = pd.read_csv('tovar_moving.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df.sort_index()

# Разделение на обучающую и тестовую выборки
train = df.iloc[:-1]
test = df.iloc[-1:]

print(f"Размер обучающей выборки: {len(train)}")
print(f"Размер тестовой выборки: {len(test)}")
print(f"Тестовое значение: {test['qty'].values[0]}")

# 2. Анализ временного ряда на наличие тренда и сезонности
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['qty'])
plt.title('Временной ряд товарооборота')
plt.xlabel('Дата')
plt.ylabel('Количество заказов')
plt.grid(True)
plt.show()

# Дополнительный анализ сезонности
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(train['qty'], model='additive', period=30)
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.show()

# 3. Прогноз с помощью экспоненциального сглаживания (α = 0.7)
model_ses = SimpleExpSmoothing(train['qty']).fit(smoothing_level=0.7, optimized=False)
forecast_ses = model_ses.forecast(1)

print(f"Прогноз экспоненциального сглаживания: {forecast_ses.values[0]:.2f}")
print(f"Фактическое значение: {test['qty'].values[0]:.2f}")
print(f"Ошибка прогноза: {abs(forecast_ses.values[0] - test['qty'].values[0]):.2f}")


# 4. Проверка ряда на стационарность
def check_stationarity(timeseries):
    # Тест Дики-Фуллера
    result = adfuller(timeseries)
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.4f}')

    if result[1] <= 0.05:
        print("Ряд стационарен (отвергаем H0)")
        return True
    else:
        print("Ряд нестационарен (не отвергаем H0)")
        return False


print("Проверка стационарности исходного ряда:")
is_stationary = check_stationarity(train['qty'])

# Если ряд нестационарен, определяем порядок интегрирования
order_integration = 0
if not is_stationary:
    # Первая разность
    diff_1 = train['qty'].diff().dropna()
    print("\nПроверка стационарности после первого дифференцирования:")
    if check_stationarity(diff_1):
        order_integration = 1
    else:
        # Вторая разность
        diff_2 = diff_1.diff().dropna()
        print("\nПроверка стационарности после второго дифференцирования:")
        if check_stationarity(diff_2):
            order_integration = 2

print(f"Порядок интегрирования: {order_integration}")

# 5. Определение порядка модели AR по графику частичной автокорреляции
plt.figure(figsize=(12, 6))
plot_pacf(train['qty'], lags=20, ax=plt.gca())
plt.title('Частная автокорреляционная функция (PACF)')
plt.show()

# Анализируем PACF для определения порядка p
# Порядок AR модели соответствует количеству значимых лагов в PACF

# 6. Построение модели AR и прогнозирование
# Определяем порядок модели (например, по количеству значимых лагов в PACF)
p = 5  # Это значение нужно определить по графику PACF

# Строим модель AR
model_ar = AutoReg(train['qty'], lags=p)
model_ar_fitted = model_ar.fit()

# Прогнозирование
forecast_ar = model_ar_fitted.forecast(steps=1)

print(f"Прогноз модели AR({p}): {forecast_ar.values[0]:.2f}")
print(f"Фактическое значение: {test['qty'].values[0]:.2f}")
print(f"Ошибка прогноза: {abs(forecast_ar.values[0] - test['qty'].values[0]):.2f}")

# 7. Сравнение результатов
results_comparison = pd.DataFrame({
    'Метод': ['Экспоненциальное сглаживание', f'AR({p})'],
    'Прогноз': [forecast_ses.values[0], forecast_ar.values[0]],
    'Фактическое': [test['qty'].values[0], test['qty'].values[0]],
    'Ошибка': [abs(forecast_ses.values[0] - test['qty'].values[0]),
               abs(forecast_ar.values[0] - test['qty'].values[0])]
})

print("\nСравнение результатов:")
print(results_comparison)

# Визуализация прогнозов
plt.figure(figsize=(12, 6))
plt.plot(train.index[-30:], train['qty'].values[-30:], label='Исторические данные')
plt.axvline(x=train.index[-1], color='gray', linestyle='--', alpha=0.7)
plt.scatter(test.index, test['qty'], color='red', label='Фактическое значение', zorder=5)
plt.scatter(test.index, forecast_ses, color='blue', label='Прогноз эксп. сглаживания', zorder=5)
plt.scatter(test.index, forecast_ar, color='green', label=f'Прогноз AR({p})', zorder=5)
plt.title('Сравнение методов прогнозирования')
plt.xlabel('Дата')
plt.ylabel('Количество заказов')
plt.legend()
plt.grid(True)
plt.show()