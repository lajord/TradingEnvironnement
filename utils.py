import plotly.graph_objects as go
import pandas as pd

df = pd.read_csv(r'H:\Desktop\Environement_Trading_Developement\ohlc_D1.csv')

fig = go.Figure(data=[go.Candlestick(
    x=df['Datetime'],
    open=df['open'], high=df['high'],
    low=df['low'], close=df['close'],
    increasing_line_color= 'cyan', decreasing_line_color= 'gray'
)])

fig.show()