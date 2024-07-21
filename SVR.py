# SVR
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import matplotlib.pyplot as plt

file_path = '/content/drive/MyDrive/temp/train27303.csv'
time_step = 24
training_rate = 0.8

def plot(data, y_test, test_predict, train_size, time_step = time_step):
    mse = mean_squared_error(y_test, test_predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, test_predict)

    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'R²: {r2}')

    plt.figure(figsize=(15, 7))

    start_date = pd.to_datetime('2015-12-23')
    end_date = pd.to_datetime('2015-12-26')
    mask = (data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)
    filtered_data = data.loc[mask]

    test_indices = np.where(mask.values)[0]
    filtered_y_test = y_test[test_indices - train_size - time_step + 1]
    filtered_test_predict = test_predict[test_indices - train_size - time_step + 1]

    plt.plot(filtered_data['timestamp'], filtered_y_test, label='Real Traffic Count', color='red')
    plt.plot(filtered_data['timestamp'], filtered_test_predict, label='Predicted Traffic Count', color='blue')
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
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]

    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_train, y_train)

    test_predict = model.predict(X_test)

    test_predict = scale.inverse_transform(test_predict.reshape(-1, 1))
    y_test = scale.inverse_transform(y_test.reshape(-1, 1))

    return data, y_test, test_predict, train_size

if __name__ == '__main__':
    data, y_test, test_predict, train_size = train(file_path)
    plot(data, y_test, test_predict, train_size)
