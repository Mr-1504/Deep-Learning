import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('train27303.csv')
print(data)
plt.figure(figsize=(20, 10))
plt.plot(data['timestamp'], data['hourly_traffic_count'], label='hourly_traffic_count', color='green')
plt.xlabel('Time')
plt.ylabel('Traffic count')
plt.title('Hourly traffic count overtime')
plt.legend()
plt.savefig('temp.png')
plt.close()
