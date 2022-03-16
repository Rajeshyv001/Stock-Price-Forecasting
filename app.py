import pandas_datareader.data as pdr
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')

start = '2010-01-01'
end = '2021-12-31'

st.title('Stocks Price Prediction')

stocks =("AAPL","GOOG","MSFT","GME","TSLA")
selected_stock = st.selectbox("Select stock ticker for prediction", stocks)

# user_input = st.text_input('Enter any stock ticker name', 'AAPL')

#scraping data from yahoo finance
data_load = st.text("Load data ...")

df = pdr.DataReader(selected_stock,'yahoo', start,end)
data_load.text("Loading data ....done!")

#Data
st.subheader('Data from 2010 - 2021')
st.write(df.tail())

# visualization
st.subheader('Closing price Vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df['Close'])
plt.title('Closing Price history')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price USD ($)', fontsize=12)
st.pyplot(fig)



#moving averages
mov_avg_day = [10,30,60]

for ma in mov_avg_day:
  colummn_name = f"MA for {ma} days"
  stock_data = df
  stock_data[colummn_name] = stock_data['Adj Close'].rolling(ma).mean()

#plot all 3 movingaverages along with closing price
st.subheader('Moving Averages')
fig=plt.figure(figsize=(12,4))
plt.plot(stock_data['Close'], label='Closing Price')
plt.plot(stock_data['MA for 10 days'], label='10 day Moving Average')
plt.plot(stock_data['MA for 30 days'], label='30 day Moving Average')
plt.plot(stock_data['MA for 60 days'], label='60 day Moving Average')
plt.title('Stock Closing Price and Moving Averages')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

#viusalising Gaussian distribution
st.subheader('Daily Return')
stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()
fig = sns.displot(stock_data['Daily_Return'].dropna(), bins=100, color='goldenrod', height=3, aspect=23/6)
plt.title('Daily Return')
st.pyplot(fig)

#Create training and test datasets
#create new dataframe with only the Close column
data = stock_data.filter(['Close'])

#convert the dataframe to a numpy array - (2D)
data_close = data.values

#get the number of rows to train the model 
# (80% train data , 20% test data)
train_size = int(np.ceil(len(data_close) * 0.80))

# scaling price value between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data_close)

#load my model
LSTM_model = load_model('Lstm_model.h5')

#Create the testing data set 
#Create a new array containing scaled values 

#from last 60 values of train to last value
test_data = scaled_data[train_size - 60 : , :]

#Create dataset x_test and y_test

x_test =[]
# for y_test can take original close price which was in numpy array form(so we don't need to inverse scaling)
y_test = data_close[train_size:, :]
for i in range (60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

#converting to numpy array
x_test = np.array(x_test)

#reshaping the data in 3D form
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#get model predicticted price
predictions = LSTM_model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# to remove pandas warning on copy of a slice 
pd.options.mode.chained_assignment =None

# plot data 
st.subheader('Stock Price Forecast')
train = data[:train_size]
validation = data[train_size :]
validation['Predictions'] = predictions
#visualize the data
fig=plt.figure(figsize=(13,5))
plt.title('Stock Price Forecast LSTM Model')
plt.xlabel('Date', fontsize = 13)
plt.ylabel('Close Price USD ($)', fontsize=13)
plt.plot(train['Close'])
plt.plot(validation[['Close', 'Predictions']])
plt.legend(['Train','Validation', 'Predictions'], loc='upper left')
st.pyplot(fig)
