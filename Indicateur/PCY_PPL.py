import numpy as np
import pandas as pd
import talib


"""
Un indicateur technique qui te permet de venir déterminer les cycles de marché
On va venir estimer mathématiquement les dynamiques sous forme de waves
"""



def compute_pcy(close, period_indicateur, period_ma_low, period_ma_high, facteur_lissage):

    ma_fast = talib.SMA(close, timeperiod=period_ma_low)  # Rapide (période courte)
    ma_slow = talib.SMA(close, timeperiod=period_ma_high)  # Lente (période longue)

    diff = ma_fast - ma_slow  
    

    min_diff = diff.rolling(window=period_indicateur).min()
    max_diff = diff.rolling(window=period_indicateur).max()


    delta_i = (diff - min_diff) / (max_diff - min_diff)


    PCY = np.zeros(len(close))


    for idx in range(1, len(close)):  
        if not np.isnan(delta_i.iloc[idx]): 
            PCY[idx] = PCY[idx-1] + facteur_lissage * (delta_i.iloc[idx] - PCY[idx-1])

    return PCY * 100



def compute_ppl(pcy,close,period_indicateur):


    min_diff = close.rolling(window=period_indicateur).min()
    max_diff = close.rolling(window=period_indicateur).max()
    PPL = pcy * (max_diff - min_diff) / 100 + min_diff
    return PPL

    


















