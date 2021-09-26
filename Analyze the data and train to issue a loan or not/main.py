import mglearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,  PolynomialFeatures

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 300)
pd.options.display.expand_frame_repr = False


g_data = pd.read_csv("C:/Users/User/Desktop/ml/GermanData.csv", names = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7',
                                              'a8', 'a9', 'a10', 'a11','a12', 'a13', 'a14',
                                              'a15', 'a16', 'a17', 'a18', 'a19', 'a20', 'status'])

# types = g_data.dtypes
# print("Number categorical featues:", sum(types=='object'))
# print(types)
#
# dis = g_data.describe()

print(g_data.head, '\n')

categorical_columns = []
for (columnName, columnData) in g_data.iteritems():
  if g_data[columnName].dtype != np.int64 and g_data[columnName].dtype != np.int8:
    categorical_columns.append(columnName)
    g_data[columnName] = g_data[columnName].astype('category').cat.codes
features_raw = g_data.drop('status', axis=1)
expected = g_data['status']

scaler = MinMaxScaler()
# I am scaling it because want everything starts with 1 (it will be more precise) and ends with max(column_name) - min(column_name)
scaler.fit(features_raw)
features = pd.DataFrame(scaler.transform(features_raw), columns=features_raw.columns)

print(g_data.head)


# X, y = g_data['a13'], g_data['a5']
# plt.plot(X, y, 'o')
#
# plt.xlabel("Признак")
# plt.ylabel("статус")
#
# plt.show()

# # with x = a2, y = a5
# X, y = g_data['a2'], g_data['a5']
# plt.plot(X, y, 'o')
#
# plt.xlabel("Признак")
# plt.ylabel("статус")
#
# plt.show()

norm_data = g_data[["a2", 'a5', 'a13', 'status']]

X, y = g_data, g_data
X_train, X_test, y_train, y_test = train_test_split(X, y , random_state=0)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

print("Прогнозы на тестовом наборе: {}".format(clf.predict(X_test)))

# print("Правильность на тестовом наборе: {:.2f}".format(clf.score(X_test, y_test)))