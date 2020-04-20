import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sma
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import autocorrelation_plot

# load data
data = pd.read_csv('Sunspots_reformat.csv', parse_dates=[0], header=0, squeeze=True)
data['Date'] = pd.to_datetime(data['Date'],  infer_datetime_format=True)
data = data.set_index(['Date'])
plt.rcParams["figure.figsize"] = [15,10]
plt.rcParams["font.size"] = 14
data.plot(y='Monthly Mean Total Sunspot Number')
plt.show()

values = data.values
rec_num = len(values)


result = seasonal_decompose(data.interpolate(), model='additive')
result.plot()
plt.show()

# autocorrelation
autocorrelation_plot(data.values)
plt.show()


# split data
split = int(rec_num*0.7)
train_vals = values[:split]
test_vals = values[split:len(values)]


# ARIMA
data = sma.datasets.sunspots.load_pandas().data
data.index = pd.Index(sma.tsa.datetools.dates_from_range('1700', '2008'))
del data["YEAR"]
model = ARIMA(data, order=(5, 1, 2)).fit(disp=False)
model.summary()
fig, ax = plt.subplots(figsize=(15, 10))
ax = data.loc['1945':].plot(ax=ax)
model.plot_predict('2000', '2020', dynamic=True, ax=ax, plot_insample=False)
plt.show()