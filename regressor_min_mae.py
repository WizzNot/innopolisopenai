import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

a = pd.read_csv(input())
features = list(filter(lambda x: x not in ("ID", "Value", "is_test"), a.columns))
df_train = a[a['is_test'] == 0]
df_train.dropna(inplace=True)
df_test = a[a['is_test'] == 1]
min_MAE = 99999999999999
best_k = 0
for k in range(1, 16):
    df_test = a[a['is_test'] == 1]
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(df_train[features], df_train['Value'])
    df_test['Value_predicted'] = model.predict(df_test[features])
    MAE = round(sum(abs(df_test['Value_predicted'] - df_test['Value'])) / len(df_test))
    if MAE < min_MAE:
        min_MAE = MAE
        best_k = k
print(best_k)
print(min_MAE)

