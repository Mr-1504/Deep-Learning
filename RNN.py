# RCNN
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten
import matplotlib.pyplot as plt

file_path = '/content/drive/MyDrive/temp/train27303.csv'
time_step = 24
train_epoch = 2
batch_size = 32
training_rate = 0.8


def plot(data, y_test, test_predict, train_size):
    mse = mean_squared_error(y_test, test_predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, test_predict)

    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'R²: {r2}')

    plt.figure(figsize=(15, 7))
    plt.plot(data['timestamp'][train_size:len(data) - time_step - 1], y_test, label='Real Traffic Count', color='red')
    plt.plot(data['timestamp'][train_size:len(data) - time_step - 1], test_predict, label='Predicted Traffic Count',
             color='blue')
    plt.xlabel('Time')
    plt.ylabel('Traffic Count')
    plt.title('Traffic Prediction')
    plt.legend()
    plt.show()


def read_data(file_path):
    data = pd.read_csv(file_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data[data['hourly_traffic_count'] != 0]
    data.sort_values('timestamp', inplace=True)

    traffic_data = data['hourly_traffic_count'].values

    scale = MinMaxScaler(feature_range=(0, 1))
    traffic_data = scale.fit_transform(traffic_data.reshape(-1, 1))
    return data, traffic_data, scale


def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)


def train(file_path):
    data, traffic_data, scale = read_data(file_path)
    X, y = create_dataset(traffic_data, time_step)

    train_size = int(len(X) * training_rate)
    test_size = len(X) - train_size
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(time_step, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=train_epoch, batch_size=batch_size, verbose=1)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = scale.inverse_transform(train_predict)
    test_predict = scale.inverse_transform(test_predict)
    y_train = scale.inverse_transform(y_train.reshape(-1, 1))
    y_test = scale.inverse_transform(y_test.reshape(-1, 1))

    return data, y_test, test_predict, train_size


if __name__ == '__main__':
    data, y_test, test_predict, train_size = train(file_path)
    plot(data, y_test, test_predict, train_size)
