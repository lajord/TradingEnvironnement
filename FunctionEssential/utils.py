import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd





"""
Precondition : Un dataframe OHLC,  un dataframe avec au moins un side c'est a dire entries et short
Postcondition : print un graphique movible
"""
def print_trades(entries_long,entries_short,exits_long,exits_short,data):
    if 'Datetime' not in data.columns:
        data['Datetime'] = data['Date']+ ' ' + data['time']

    fig = go.Figure(data=[go.Candlestick(
        x=data['Datetime'],
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        increasing_line_color='white',
        decreasing_line_color='red'
    )])
    fig.add_trace(go.Scatter(
        x=data['Datetime'][entries_long],
        y=data['close'][entries_long],
        mode='markers',
        marker=dict(symbol='x', size=10, color='blue'),
        name='Entrée Long'
    ))
    fig.add_trace(go.Scatter(
        x=data['Datetime'][entries_short],
        y=data['close'][entries_short],
        mode='markers',
        marker=dict(symbol='x', size=10, color='red'),
        name='Entrée Short'
    ))
    fig.add_trace(go.Scatter(
        x=data['Datetime'][exits_long],
        y=data['close'][exits_long],
        mode='markers',
        marker=dict(symbol='triangle-up', size=10, color='blue'),
        name='Sortie Long'
    ))
    fig.add_trace(go.Scatter(
        x=data['Datetime'][exits_short],
        y=data['close'][exits_short],
        mode='markers',
        marker=dict(symbol='triangle-down', size=10, color='red'),
        name='Sortie Short'
    ))

    fig.update_layout(
        title="Visualisation des Trades",
        xaxis_title="Date/Heure",
        yaxis_title="Prix",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.show()





def get_pnl(portfolio):
    equity = portfolio.value()
    equity.vbt.plot().show()


def get_mean_pnl_multi_asset(portfolio):
    all_values = portfolio.value()  
    mean_pnl = all_values.mean(axis=1) 
    plt.figure(figsize=(12, 6))
    plt.plot(mean_pnl, label="Courbe PnL Moyenne (Lissée)")
    plt.xlabel("Temps")
    plt.ylabel("Valeur Moyenne du Portefeuille")
    plt.title("Performance Moyenne du Portefeuille Multi-Actifs")
    plt.legend()
    plt.show()