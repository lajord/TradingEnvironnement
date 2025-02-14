import pandas as pd
import numpy as np
import plotly.graph_objects as go

"""
LAW OF REVERSION
Source : Statistical Arbitrage Andrew Pole 

L'objectif de cette indicateur c'est d'essayer d'isoler les périodes de range / chop / mean reversion des périodes de tendances 
L'indicateur ce base sur la règle des 75% expliquer dans le livre

l'indicateur aura une seuil au 75 et une valeur entière possitive comprise entre 0 et 100 inclus 

Fenetre loockback = 150

"""


def adaptive_market_regime(close, windows=150, beta=5):
    if isinstance(close, np.ndarray):
        close = pd.Series(close.squeeze())


    spreads = close.diff().dropna()
    moving_avg = spreads.ewm(span=windows//2, adjust=False).mean()
    market_regime_value = []

    # Boucle principale
    for i in range(windows * 2, len(close)):  
        window_data = close[i - windows:i]
        spreads = window_data.diff().dropna()
        current_moving_avg = moving_avg[i - windows:i].dropna()
        

        verification = 0
        for j in range(1, len(spreads)):  
            if spreads.iloc[j - 1] > spreads.iloc[j]:
                verification += 1 if spreads.iloc[j - 1] > current_moving_avg.iloc[j] else 0
            else:
                verification += 1 if spreads.iloc[j - 1] < current_moving_avg.iloc[j] else 0

  
        proba = 100 / (1 + np.exp(-beta * ((verification / windows) - 0.5)))
        market_regime_value.append(proba)

    # Ajouter des NaN initiaux
    market_regime_value = [np.nan] * (windows * 2) + market_regime_value

    # Transformer en Series et lisser
    market_regime_smoothed = pd.Series(market_regime_value).rolling(window=5, min_periods=1).mean()
    
    return market_regime_smoothed


