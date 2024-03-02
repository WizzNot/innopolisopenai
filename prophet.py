from prophet import Prophet
import pandas as pd

m = Prophet()
a = pd.read_csv("airplanes.csv")
a['Activity Period'] = pd.to_datetime(a['Activity Period'], format='%Y-%m-%d')
a = a.rename(columns={"Activity Period": "ds", "Passenger Count": "y"}).reset_index()
m.fit(a)
future = m.make_future_dataframe(periods=12, freq="MS", include_history=False)
forecast = m.predict(future)
otv = forecast[['ds', 'yhat']].rename(columns={"ds": "Activity Period", "yhat": "Passenger Count"})
otv.to_csv('output.csv', index=False)