import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler,
    LabelEncoder,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             confusion_matrix,
                             roc_curve,
                             roc_auc_score,
                             )

# Modeling
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)
from sklearn.model_selection import (
    train_test_split,  # used to split the data into training and testing
    RandomizedSearchCV,  # used for tuning the models parameters
    cross_val_score,
)

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

traffic_df = pd.read_csv('/content/drive/MyDrive/temp/Traffic - Copy.csv')
traffic_df.head()

traffic_df.describe()

traffic_df.columns

traffic_df.dtypes

df_complete = traffic_df.copy()
df_1d = traffic_df.head(96)  # it need in EDA latter!
df_1d

traffic_df['midday'] = ''

for i in range(len(traffic_df['Time'])):

    if traffic_df['Time'][i][-2:] == 'AM':
        traffic_df.loc[i, 'midday'] = 'AM'

    elif traffic_df['Time'][i][-2:] == 'PM':
        traffic_df.loc[i, 'midday'] = 'PM'

# removing 'AM' or 'PM' form Time column        
traffic_df['Time'] = traffic_df['Time'].str[:-2]

traffic_df

# The theme and size and resolution of the plots
# figure size
plt.figure(figsize=(6, 3))
# background, color type
sns.set_theme(style="whitegrid", palette="muted")
sns.set_context('notebook', font_scale=1.0, rc={"lines.linewidth": 1.5})
# font properties
plt.rcParams['font.family'] = 'cursive'
plt.rcParams['font.stretch'] = 'condensed'
plt.rcParams['font.style'] = 'italic'
plt.rcParams['font.weight'] = 'heavy'
plt.rcParams['font.size'] = 15
# resolution
plt.rcParams['figure.dpi'] = 120  # resolution

# Plot the numeric features
numeric_columns = ['Time', 'Date', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total']
sns.pairplot(traffic_df[numeric_columns])
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
fig.suptitle('Effects of vehicles')
# ------------- First-plot(1,1)
sns.histplot(traffic_df, x='CarCount', hue='Traffic Situation', kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Effect of Car')
# ------------- Second-plot(1,2)
sns.histplot(traffic_df, x='BikeCount', hue='Traffic Situation', kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Effect of bike')
# ------------- Third-plot(2,1)
sns.histplot(traffic_df, x='BusCount', hue='Traffic Situation', kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Effect of Bus')
# ------------ Forth-plot(2,2)
sns.histplot(traffic_df, x='TruckCount', hue='Traffic Situation', kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Effect of Truck 🚚 ')
plt.subplots_adjust(hspace=0.3)
plt.show()

hours = ['12:00', '', '', '', '1:00', '', '', '', '2:00', '', '', '', '3:00', '', '', '', '4:00', '', '', '', '5:00',
         '', '', '', '6:00', '', '', '', '7:00', '', '', '', '8:00', '', '', '', '9:00', '', '', '', '10:00', '', '',
         '', '11:00']

fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
fig.suptitle('Time')

sns.histplot(traffic_df, x='Time', hue='Traffic Situation', kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Time & Traffic Situation')
axes[0, 0].set_xticklabels(hours, rotation=45)

sns.histplot(traffic_df, x='Date', hue='Traffic Situation', kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Date & Traffic Situation')

sns.countplot(data=traffic_df, x="Day of the week", hue="Traffic Situation", palette="Set2", ax=axes[1, 0])
axes[1, 0].set_xlabel("Day of the week")
axes[1, 0].set_ylabel("Count")
axes[1, 0].set_title("Traffic Situation by Day of the Week")
# Set the tick positions and labels
tick_positions = range(len(traffic_df['Day of the week'].unique()))
tick_labels = traffic_df['Day of the week'].unique()

# Set the tick positions and labels on the x-axis
axes[1, 0].set_xticks(tick_positions)
axes[1, 0].set_xticklabels(tick_labels, rotation=45)
# axes[1, 0].set_xticklabels(traffic_df['Day of the week'], rotation=45)

sns.histplot(traffic_df, x='Total', hue='Traffic Situation', kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Total vehicle & Traffic Situation')

plt.subplots_adjust(hspace=0.5)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
fig.suptitle('Amount of Vehicle')
# -------------- First Plot (1,1)
axes[0].plot(df_1d['Time'], df_1d['CarCount'])
axes[0].plot(df_1d['Time'], df_1d['BikeCount'])
axes[0].legend(df_complete.columns[3:5])
axes[0].set_title('Time vs Car and bike count')
axes[0].set_xticks(df_1d['Time'][::10])  # Display every 10th label
axes[0].set_xticklabels(df_1d['Time'][::10], rotation=45)

# ------------- Second Plot (1,2)
axes[1].plot(df_1d['Time'], df_1d['BusCount'])
axes[1].plot(df_1d['Time'], df_1d['TruckCount'])
axes[1].set_title('Time vs Bus and Truck count')
axes[1].legend(df_complete.columns[5:7])
axes[1].set_xticks(df_1d['Time'][::10])  # Display every 10th label
axes[1].set_xticklabels(df_1d['Time'][::10], rotation=45)

plt.show()

# Plot the target variable 'Traffic Situation'
sns.countplot(x='Traffic Situation', data=traffic_df)
plt.show()

# Assuming 'Time' is the name of the column in your DataFrame
traffic_df['Time'] = pd.to_datetime(traffic_df['Time']).dt.hour * 3600 + \
                     pd.to_datetime(traffic_df['Time']).dt.minute * 60 + \
                     pd.to_datetime(traffic_df['Time']).dt.second
traffic_df.head()

traffic_df['Traffic Situation'].value_counts()

# Separate the features and target variable
features = traffic_df.drop(['Traffic Situation'], axis=1)
target = traffic_df['Traffic Situation']

# Normalize the numeric features using MinMaxScaler
numeric_columns = ['Time', 'Date', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total']
scaler = MinMaxScaler()
features[numeric_columns] = scaler.fit_transform(features[numeric_columns])

# Encode the categorical feature 'midday' using LabelEncoder
le = LabelEncoder()
features['midday'] = le.fit_transform(features['midday'])
features['Day of the week'] = le.fit_transform(features['Day of the week'])

# Encode the target variable 'Traffic Situation' using LabelEncoder
le_target = LabelEncoder()
target = le_target.fit_transform(target)

# Concatenate the features and target variable
normalized_encoded_data = pd.concat([features, pd.Series(target, name='Traffic Situation')], axis=1)

traffic_df = normalized_encoded_data
traffic_df

X = traffic_df.drop('Traffic Situation', axis=1)
y = traffic_df['Traffic Situation']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

X_train
y_train

init_models = {
    'support vector linear': SVR(kernel='linear'),
    'support vector kernel': SVR(kernel='rbf'),
    'XGBOOST': XGBRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'LGBM': LGBMRegressor()
}

R2 = []
models_names = []
for i, (key, model) in enumerate(init_models.items()):
    model.fit(X_train, y_train)
    models_names.append(key)
    R2.append(np.mean(cross_val_score(model, X_train, y_train, cv=5)))
models_scores = pd.DataFrame({'model name': models_names, 'R2 score': R2})
models_scores.head(7)

fig, ax = plt.subplots()
ax.set_xticklabels(ax.get_xticklabels(), rotation=70)
ax.set_title("models R2 score")
sns.barplot(data=models_scores, x='model name', y="R2 score")
plt.show()

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_test, y_pred

y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision with 'weighted' average setting
precision = precision_score(y_test, y_pred, average='weighted')

# Calculate recall with 'weighted' average setting
recall = recall_score(y_test, y_pred, average='weighted')

# Calculate F1 score with 'weighted' average setting
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'f1_score: {f1}')
# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm


def plot_confusion_matrix(confusion_matrix, classes):
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], fmt),
                     ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    # Show the plot
    plt.show()


# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plot_confusion_matrix(cm, classes=['Heavy', 'High', 'Low', 'Normal'])
plt.show()
