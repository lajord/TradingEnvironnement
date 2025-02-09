import pandas as pd
import numpy as np
import vectorbt as vbt




def calculate_vpn(high, low, close, volume, period):

    # Vérifier que les dimensions des entrées sont cohérentes
    if not (high.shape == low.shape == close.shape == volume.shape):
        raise ValueError("Les tableaux 'high', 'low', 'close' et 'volume' doivent avoir les mêmes dimensions.")
    
    if high.ndim == 1:
        high = high[:, np.newaxis]
        low = low[:, np.newaxis]
        close = close[:, np.newaxis]
        volume = volume[:, np.newaxis]
    
    value_vpn = np.empty_like(high, dtype=np.float64)
    
    for i in range(high.shape[1]):
        # Convertir les colonnes en Pandas Series avec un index temporel (optionnel)
        high_series = pd.Series(high[:, i])
        low_series = pd.Series(low[:, i])
        close_series = pd.Series(close[:, i])
        volume_series = pd.Series(volume[:, i])
        
        # Calculer l'ATR avec VectorBT
        atr_result = vbt.ATR.run(high_series, low_series, close_series, window=period)
        atr = atr_result.atr
        
        # Calculer le prix typique
        type_price = (high_series + low_series + close_series) / 3
        
        # Calculer la distance en utilisant ATR décalé
        distance = 0.1 * atr.shift(1).fillna(0)
        
        # Initialiser les séries Vp et Vn
        Vp = pd.Series(0, index=type_price.index)
        Vn = pd.Series(0, index=type_price.index)
        
        # Calculer Vp et Vn
        condition_vp = type_price > (type_price.shift(1) + distance)
        condition_vn = type_price < (type_price.shift(1) - distance)
        
        Vp[condition_vp] = volume_series[condition_vp]
        Vn[condition_vn] = volume_series[condition_vn]
        
        # Calcul des sommes glissantes
        Vp_sum = Vp.rolling(window=period).sum()
        Vn_sum = Vn.rolling(window=period).sum()
        Vtotal_sum = volume_series.rolling(window=period).sum().replace(0, np.nan)  # Eviter la division par zéro
        
        # Calcul du VPN brut
        VPN_raw = 100 * (Vp_sum - Vn_sum) / Vtotal_sum
        
        # Lissage EMA
        VPN = VPN_raw.ewm(span=2, adjust=False).mean()
        
        # Remplir la colonne VPN dans le tableau de résultats
        value_vpn[:, i] = VPN.values
    
    return value_vpn


