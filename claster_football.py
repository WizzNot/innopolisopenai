# Отличаются ли футболисты, играющие на разных позициях, по своим характеристикам?
# Вам дан уже знакомый датасет по футбольным игрокам footbal_players_clust.csv. Кластеризуйте его по набору признаков features, используя алгоритм kmeans со следующими параметрами:
#    число кластеров: 10
#    random_state: 123
# Формат ввода
# На вход функции process(df) поступает исходный датафрейм.
# Формат вывода
# Функция должна вернуть строку, содержащую значение Best Position, соответствующее самой популярной позиции в самом большом кластере.


import pandas as pd
from sklearn.cluster import KMeans
features = ['Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys','Dribbling','Curve','FKAccuracy','LongPassing','BallControl','Acceleration','SprintSpeed','Agility','Reactions','Balance','ShotPower','Jumping','Stamina','Strength','LongShots','Aggression','Interceptions','Positioning','Vision','Penalties','Composure']


def process(df):
    kmeans = KMeans(n_clusters=10, random_state=123)
    df['cluster'] = kmeans.fit_predict(df[features])
    largest_cluster = df['cluster'].value_counts().idxmax()
    most_common_position = df[df['cluster'] == largest_cluster]['Best Position'].value_counts().idxmax()
    return most_common_position