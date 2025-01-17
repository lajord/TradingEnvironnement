import numpy as np
import pandas as pd 
import vectorbt as vbt
import talib 
import matplotlib.pyplot as plt 
import VPN as vpn

import sys
import os
chemin_parent = os.path.abspath(r'C:\Users\Jordi\Desktop\Trading_Dev_Stratégie_Environement\FunctionEssential')

if chemin_parent not in sys.path:
    sys.path.append(chemin_parent)

import Trend_following as tf

import function_essential as fe


list_tickers = ["NK_TOTAL_4H"]

cycles = fe.split(list_tickers, "2013-01-01", "2024-01-01", 1, 5)
capital_initial = 10000

print(cycles)


strategie = tf.VpnFollow(0,"240m",0)
all_backtest = []
equity = 0
for ticker in list_tickers:
    nb_cycle = len(cycles[ticker]) + 1
    print(nb_cycle)
    strategie.set_tickers([ticker])
    for cycle_index in range(1,nb_cycle):
        print(cycle_index)
        IOS = cycles[ticker][f'cycle_{cycle_index}']['IOS']
        OOS = cycles[ticker][f'cycle_{cycle_index}']['OOS']
        strategie.set_data(IOS)

        param1 = np.arange(5,14,1)
        param2 = np.arange(10,21,1)
        params = strategie.optimize(3,4,2,2,param1,param2,30,30,30,15)
        colonne_max = params.idxmax(axis=1).iloc[0]
        strategie.set_data(OOS)
        backtest = strategie.backtest(capital_initial,0,0,2,2,colonne_max[0],colonne_max[1],30,30,30,15)
        all_backtest.append(backtest)
        capital_initial = backtest.total_profit() + capital_initial
        equity = equity + backtest.value() 
        fe.get_pnl(backtest)


plt.figure(figsize=(10, 6))
equity.plot(title="Courbe d'équité cumulée", ylabel="Valeur totale", xlabel="Temps")
plt.show()