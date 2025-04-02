###Import####
import numpy as np
import pandas as pd 
import vectorbt as vbt
import talib 
import matplotlib.pyplot as plt 
import VPN as vpn
import MetaTrader5 as mt5

import sys
import os
chemin_parent = os.path.abspath(r'C:\Users\Jordi\Desktop\Trading_Dev_Stratégie_Environement\FunctionEssential')

if chemin_parent not in sys.path:
    sys.path.append(chemin_parent)

import function_essential as fe


######################################################################################################################################################################################

def create_VPN(high,low,close,volume,period):
    high = high.to_numpy()
    low = low.to_numpy()
    close = close.to_numpy()
    volume = volume.to_numpy()
    return vpn.calculate_vpn(high,low,close,volume,period)

def high_max(high, period):
    max_values = np.empty_like(high)
    for i in range(high.shape[1]):
        max_values[:, i] = talib.MAX(high[:, i], timeperiod=period)
    return max_values


def create_span_1(high, low, period_base, period_conv):
    spanA_result = np.empty_like(high)
    for i in range(high.shape[1]):
        conversion_max = talib.MAX(high[:, i], timeperiod=period_conv)
        conversion_min = talib.MIN(low[:, i], timeperiod=period_conv)
        conversion_line = (conversion_max + conversion_min) / 2
        base_max = talib.MAX(high[:, i], timeperiod=period_base)
        base_min = talib.MIN(low[:, i], timeperiod=period_base)
        base_line = (base_max + base_min) / 2
        leading_span_a = (conversion_line + base_line) / 2
        spanA_result[:, i] = leading_span_a
    return spanA_result

def create_span_spread(high, low, period_base, period_conv,spread):
    spanA_result = np.empty_like(high)
    for i in range(high.shape[1]):
        conversion_max = talib.MAX(high[:, i], timeperiod=period_conv)
        conversion_min = talib.MIN(low[:, i], timeperiod=period_conv)
        conversion_line = (conversion_max + conversion_min) / 2
        base_max = talib.MAX(high[:, i], timeperiod=period_base)
        base_min = talib.MIN(low[:, i], timeperiod=period_base)
        base_line = (base_max + base_min) / 2
        leading_span_a = (conversion_line + base_line) / 2
        spanA_1_series = pd.Series(leading_span_a)
        spanA_26 = spanA_1_series.shift(spread)
        spanA_result[:, i] = spanA_26
    return spanA_result



################################################################################################################################################################


def create_entry_signal(close,spanA_1, spanA_26, VPN,SMA_30,seuil,nb_bougie):
    
    conditions = (
        # (spanA_1 > spanA_26) &
        (VPN > seuil) &
        (close > SMA_30)
    )
    entries = np.zeros_like(close, dtype=bool)
    entry_indices = np.where(conditions)[0]
    last_entry_index = -nb_bougie
    for index in entry_indices:
        if index > last_entry_index + nb_bougie:
            entries[index] = True
            last_entry_index = index
    
    return entries

def create_exit_signals(close, entries, VPN_MA30, VPN, High_5, ATR_14,nb_bougie):
    result_exits = np.zeros_like(close, dtype=bool)
    

    for j in range(close.shape[1]):
        exits = np.zeros(close.shape[0], dtype=bool)
        for entry_index in np.where(entries[:, j])[0]:

            for offset in range(1, nb_bougie+1):
                i = entry_index + offset
                if i >= len(close): 
                    break

                if (VPN[i, j] < VPN_MA30[i, j]) and (close[i, j] < High_5[i, j] - 3 * ATR_14[i, j]):
                    exits[i] = True
                    break
            else:

                if entry_index + nb_bougie < len(close):
                    exits[entry_index + nb_bougie] = True
        
        result_exits[:, j] = exits
    
    return result_exits


#############################################################################################################################################################

"""
Création des IndicatorFactory

"""


Ind_entries = vbt.IndicatorFactory(
    class_name="create_entry_signal",
    input_names=["close", "VPN","spanA_1","spanA_26","SMA_30"],
    param_names=["seuil","nb_bougie"],
    output_names=["entries"]
).from_apply_func(create_entry_signal)

Ind_exits_edge = vbt.IndicatorFactory(
    class_name="create_exit_signals",
    input_names=["close","entries", "VPN_MA30", "VPN","High_5","ATR_14"],
    param_names=["nb_bougie"],
    output_names=["exits"]
).from_apply_func(create_exit_signals)

Ind_spanA_1 = vbt.IndicatorFactory(
    class_name="create_span_1",
    input_names=["high","low"],
    param_names=["period_base","period_conv"],
    output_names=["spanA_1"]
).from_apply_func(create_span_1)


Ind_spanA_spread = vbt.IndicatorFactory(
    class_name="creaate_span_spread",
    input_names=["high","low"],
    param_names=["period_base","period_conv","spread"],
    output_names=["spanA_26"]
).from_apply_func(create_span_spread)

Ind_high_max = vbt.IndicatorFactory(
    class_name="high_max",
    input_names=["high"],
    param_names=["period"],
    output_names=["high"]
).from_apply_func(high_max)










class VpnFollow():

    def __init__(self,data,frequence,list_tickers):
        self.data = data
        self.frequence = frequence
        self.list_tickers = list_tickers

    def backtest(self,capital_initial,period_start,period_end,p_base,p_conv,p_high,p_atr,period_ma,p_vpn,period_vpnMA,nb_bougie):
        if (period_end == 0) and (period_start == 0):

            entries_df = pd.DataFrame()
            exits_df = pd.DataFrame()
            close_df = pd.DataFrame()

            for ticker in self.list_tickers:

                
                data = self.data[ticker]
                valeur_vpn =  create_VPN(data['high'],data['low'],data['close'],data['volume'],30)
                list_value = self.make_value_no_combi(data['high'],data['low'],data['close'],p_base,p_conv,p_high,p_atr,valeur_vpn,period_ma,period_vpnMA)
                entries = self.make_entries(data['close'],list_value,valeur_vpn,nb_bougie)
                exits = self.make_exits(data['close'],entries,list_value,valeur_vpn,nb_bougie)
                close = data['close']
                entries = entries.reset_index(drop=True)
                exits = exits.reset_index(drop=True)
                close = close.reset_index(drop=True)

                
                entries_df[ticker] = entries
                exits_df[ticker] = exits
                close_df[ticker] = close

            close_df = close_df.replace(0, None).fillna(method='ffill')
            portfolio = vbt.Portfolio.from_signals(
                close_df,
                entries_df,
                exits_df,
                freq=self.frequence,
                fees=0.0001,
                slippage=0.00005,
                init_cash=capital_initial
            )

            return portfolio

        else:
            pass

    

  
    def optimize(self,metrics,choice,p_base,p_conv,p_high,p_atr,period_ma,p_vpn,period_vpnMA,nb_bougie):
        # Utiliser template pour faire varier les optis 
        dataframe_save = {}

        if choice == 2:
            # Faire span et bougie de sortie 
            combi1 = fe.get_combinaison(p_base,p_conv)
            combi2= fe.get_combinaison(combi1,nb_bougie)
            for ticker in self.list_tickers:
                entries_dict = {}  # Dictionnaire pour stocker les DataFrames des entrées
                exits_dict = {}    # Dictionnaire pour stocker les DataFrames des sorties
                data = self.data[ticker]
                valeur_vpn = create_VPN(data['high'], data['low'], data['close'], data['volume'], p_vpn)
                list_value = self.make_value_no_combi(data['high'], data['low'], data['close'], p_base, p_conv, p_high, p_atr, valeur_vpn, period_ma, period_vpnMA)
                spanA = list_value[0]
                spanA26 = list_value[1]
                count = 0

                for i in range(spanA.shape[1]):
                    spanA_1_1D = spanA.iloc[:, i]
                    spanA_26_1D = spanA26.iloc[:, i]
                    list_value[0] = spanA_1_1D
                    list_value[1] = spanA_26_1D
                    for j in range(len(nb_bougie)):
                        # Générer les DataFrames pour entries et exits
                        entries = self.make_entries(data['close'], list_value, valeur_vpn, nb_bougie[j])
                        exits = self.make_exits(data['close'], entries, list_value, valeur_vpn, nb_bougie[j])
                        entries_dict[count] = entries
                        exits_dict[count] = exits
                        count += 1

                # Concaténer les DataFrames dans le dictionnaire en un seul DataFrame
                entries_df = pd.concat(entries_dict.values(), axis=1)
                entries_df.columns = range(entries_df.shape[1])  # Colonnes avec indices numériques successifs

                exits_df = pd.concat(exits_dict.values(), axis=1)
                exits_df.columns = range(exits_df.shape[1])
                close = data['close']
                close_df = close.replace(0, None).fillna(method='ffill')
     
                portfolio = vbt.Portfolio.from_signals(
                     close_df,
                     entries_df,
                     exits_df,
                     freq= self.frequence,
                     slippage=0.0005
                )
                df = fe.get_df_optimisation_parameter(combi2,portfolio,metrics)
                dataframe_save[ticker] = df
            df_final = fe.get_information_combinaison_optimistion(dataframe_save,combi2)
            colonne_max = df_final.idxmax(axis=1).iloc[0]
            print(f"L'index (colonne) correspondant à la valeur maximale est : {colonne_max}")
            return df_final
        

        if choice == 3:
            combi1 = fe.get_combinaison(period_vpnMA,p_atr)
            for ticker in self.list_tickers:
                entries_dict = {}  # Dictionnaire pour stocker les DataFrames des entrées
                exits_dict = {}    # Dictionnaire pour stocker les DataFrames des sorties
                data = self.data[ticker]
                valeur_vpn = create_VPN(data['high'], data['low'], data['close'], data['volume'], p_vpn)
                list_value = self.make_value_no_combi(data['high'], data['low'], data['close'], p_base, p_conv, p_high, p_atr, valeur_vpn, period_ma, period_vpnMA)
                VNPMA = list_value[3]
                ATR = list_value[4]
                count = 0

                for i in range(VNPMA.shape[1]):
                    VNPMA_1D = VNPMA.iloc[:, i]
                    list_value[3] = VNPMA_1D
                    for j in range(ATR.shape[1]):
                        ATR_1D = ATR.iloc[:, j]
                        list_value[4] = ATR_1D
                        # Générer les DataFrames pour entries et exits
                        entries = self.make_entries(data['close'], list_value, valeur_vpn, nb_bougie)
                        exits = self.make_exits(data['close'], entries, list_value, valeur_vpn,nb_bougie )
                        entries_dict[count] = entries
                        exits_dict[count] = exits
                        count += 1

                # Concaténer les DataFrames dans le dictionnaire en un seul DataFrame
                entries_df = pd.concat(entries_dict.values(), axis=1)
                entries_df.columns = range(entries_df.shape[1])  # Colonnes avec indices numériques successifs

                exits_df = pd.concat(exits_dict.values(), axis=1)
                exits_df.columns = range(exits_df.shape[1])
                close = data['close']
                close_df = close.replace(0, None).fillna(method='ffill')
     
                portfolio = vbt.Portfolio.from_signals(
                     close_df,
                     entries_df,
                     exits_df,
                     freq= self.frequence,
                     slippage=0.0005
                )
                df = fe.get_df_optimisation_parameter(combi1,portfolio,metrics)
                dataframe_save[ticker] = df
            df_final = fe.get_information_combinaison_optimistion(dataframe_save,combi1)
            colonne_max = df_final.idxmax(axis=1).iloc[0]
            print(f"L'index (colonne) correspondant à la valeur maximale est : {colonne_max}")
            return df_final
        
        if choice == 4:
            combi1 = fe.get_combinaison(p_high,p_atr)
            for ticker in self.list_tickers:
                entries_dict = {}  # Dictionnaire pour stocker les DataFrames des entrées
                exits_dict = {}    # Dictionnaire pour stocker les DataFrames des sorties
                data = self.data[ticker]
                valeur_vpn = create_VPN(data['high'], data['low'], data['close'], data['volume'], p_vpn)
                list_value = self.make_value_no_combi(data['high'], data['low'], data['close'], p_base, p_conv, p_high, p_atr, valeur_vpn, period_ma, period_vpnMA)
                HIGH = list_value[5]
                ATR = list_value[4]
                count = 0

                for i in range(HIGH.shape[1]):
                    HIGH_1D = HIGH.iloc[:, i]
                    list_value[5] = HIGH_1D
                    for j in range(ATR.shape[1]):
                        ATR_1D = ATR.iloc[:, j]
                        list_value[4] = ATR_1D
                        # Générer les DataFrames pour entries et exits
                        entries = self.make_entries(data['close'], list_value, valeur_vpn, nb_bougie)
                        exits = self.make_exits(data['close'], entries, list_value, valeur_vpn,nb_bougie )
                        entries_dict[count] = entries
                        exits_dict[count] = exits
                        count += 1

                # Concaténer les DataFrames dans le dictionnaire en un seul DataFrame
                entries_df = pd.concat(entries_dict.values(), axis=1)
                entries_df.columns = range(entries_df.shape[1])  # Colonnes avec indices numériques successifs

                exits_df = pd.concat(exits_dict.values(), axis=1)
                exits_df.columns = range(exits_df.shape[1])
                close = data['close']
                close_df = close.replace(0, None).fillna(method='ffill')
     
                portfolio = vbt.Portfolio.from_signals(
                     close_df,
                     entries_df,
                     exits_df,
                     freq= self.frequence,
                     slippage=0.0005
                )
                df = fe.get_df_optimisation_parameter(combi1,portfolio,metrics)
                dataframe_save[ticker] = df
            df_final = fe.get_information_combinaison_optimistion(dataframe_save,combi1)
            colonne_max = df_final.idxmax(axis=1).iloc[0]
            print(f"L'index (colonne) correspondant à la valeur maximale est : {colonne_max}")
            return df_final



    def set_data(self,data):
        self.data = data

    def set_tickers(self,tickers):
        self.list_tickers = tickers

    def make_entries(self,close,list_value,valeur_vpn,nb_bougie):
        Value_entries = Ind_entries.run(close,list_value[0],list_value[1],valeur_vpn,list_value[2],10,nb_bougie)
        return Value_entries.entries

    def make_exits(self,close,entries,list_value,valeur_vpn,nb_bougie):
        exits = Ind_exits_edge.run(close,entries,list_value[3],valeur_vpn,list_value[5],list_value[4],nb_bougie)
        return exits.exits

    def make_value_no_combi(self,high,low,close,p_base,p_conv,p_high,p_atr,valeur_vpn,period_ma, period_vpnMA):

        Value_spanA_1 = Ind_spanA_1.run(high,low,period_base=p_base,period_conv=p_conv,param_product = True)
        Value_spanA_26 = Ind_spanA_spread.run(high,low,period_base=p_base,period_conv=p_conv,spread=26,param_product = True)
        Value_high = Ind_high_max.run(high,period=p_high,param_product = True)
        Value_atr = vbt.ATR.run(high,low,close,window=p_atr,param_product = True)
        Value_vpnMA = vbt.MA.run(valeur_vpn, window=period_vpnMA,param_product = True)
        Value_MA = vbt.MA.run(close, window=period_ma,param_product = True)

        spanA_1 = Value_spanA_1.spanA_1
        spanA_26 = Value_spanA_26.spanA_26
        MA = Value_MA.ma
        ATR = Value_atr.atr
        high = Value_high.high
        vpnMA = Value_vpnMA.ma

        valeurs = [spanA_1, spanA_26, MA, vpnMA, ATR, high]

        return valeurs






# tickers2 = ["ym_TOTAL_4H"]
# data2 = fe.get_data(tickers2, "2024-01-01", "2024-12-31")


# strategie = VpnFollow(data2,"240m",tickers2)

# # # # # # param1 = np.arange(20,70,5)
# # # # # # param2 = np.arange(20,70,5)
# # # # # # nb_bougie = np.arange(3,20,1)


# # param1 = np.arange(5,14,1)
# # param2 = np.arange(10,21,1)
# # params = strategie.optimize(3,4,2,2,param1,param2,30,30,30,15)


# # # #def backtest(self,period_start,period_end,p_base,p_conv,p_high,p_atr,period_ma,p_vpn,period_vpnMA,nb_bougie):
# portfolio = strategie.backtest(10000,0,0,2,2,9,17,30,30,30,15)
# fe.get_pnl(portfolio)

symbols = ["EURUSD"]
data = fe.get_data_forex(symbols, timeframe=mt5.TIMEFRAME_H1)

strategie = VpnFollow(data,"240m",symbols)
portfolio = strategie.backtest(10000,0,0,2,2,9,17,30,30,30,15)
fe.get_pnl(portfolio)