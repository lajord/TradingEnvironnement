import numpy as np
import pandas as pd
import vectorbt as vbt
import matplotlib.pyplot as plt
import math

USDJPY_W1 = pd.read_csv(r'H:\Desktop\Environement_Trading_Developement\USDJPY_W1.csv')
USDJPY_D1 = pd.read_csv(r'H:\Desktop\Environement_Trading_Developement\USDJPY_D1.csv')
USDJPY_H4 = pd.read_csv(r'H:\Desktop\Environement_Trading_Developement\USDJPY_H4.csv')
USDJPY_H1 = pd.read_csv(r'H:\Desktop\Environement_Trading_Developement\USDJPY_H1.csv')
USDJPY_M30 = pd.read_csv(r'H:\Desktop\Environement_Trading_Developement\USDJPY_M30.csv')
USDJPY_M15 = pd.read_csv(r'H:\Desktop\Environement_Trading_Developement\USDJPY_M15.csv')
USDJPY = {'USDJPY_W1' : USDJPY_W1,'USDJPY_D1' : USDJPY_D1,'USDJPY_H4' :USDJPY_H4,'USDJPY_H1' : USDJPY_H1, 'USDJPY_M30' : USDJPY_M30, 'USDJPY_M15' : USDJPY_M15}




"""
Pour gérer le multiTrimeframe tu fais ca avec pandas
tu importes ta data donc soit via metatrader tu importe toute les timeframe
Soit via pandas ou tu utilise .ohlc() exemple : daily_data = data['close'].resample('1D').ohlc()

Ensuite il faut faire un dataframe global avec une colonne pour un indicateur de chaque timeframe 
-> Aligner sur les index daily_sma_aligned = daily_sma.sma.reindex(data.index, method='ffill')
ici ca permet en gros d'aligner sur les Data donc au final ca aligne correctement car quand 
il manque des valeurs alors ca fill avec la valeur précédente 

for col in df.columns: parcourir les colonnes d'un dataframe 
"""

"""
Par rapport au TDI quand tu vas calculer le RSI en fonction de la time frame derrière tu met tout sur le meme dataframe
donc tu auras un dataframe avec 10 colonnes car y'a une fast et une low pour chaque RSI et derrière tu met tout sur le meme
index et enfaite les jours avec les meme dates vont s'aligner et par exemple pour combler les trous on prend le jour précédent jusqu'a
que ca change de jour 

"""




#//////////////////////////////////////
# Creation Indicateur
#///////////////////////////////////////


def TDI(rsi,fast_period,low_period,midle_period):
    fast_ma = vbt.MA.run(rsi, window=fast_period).ma
    low_ma = vbt.MA.run(rsi, window=low_period).ma
    midle_ma = vbt.MA.run(rsi, window=midle_period).ma
    return fast_ma,low_ma,midle_ma




def point_pivot(close, high, low, nbr_pivot):
    pivot_collection = []
    pivot_support_0 = (close + high + low) / 3
    pivot_collection.append(pivot_support_0)
    pivot_resistance_0 = (close + high + low) / 3
    pivot_collection.append(pivot_resistance_0)
    pivot_support_1 = 2 * pivot_support_0 - high
    pivot_resistance_1 = 2 * pivot_support_0 - low
    pivot_collection.append(pivot_support_1)
    pivot_collection.append(pivot_resistance_1)
    for i in range(2, nbr_pivot):
        pivot_support_i = pivot_support_0 - (high - low) * (i - 1)
        pivot_resistance_i = pivot_support_0 + (high - low) * (i - 1)
        pivot_collection.append(pivot_support_i)
        pivot_collection.append(pivot_resistance_i)
    point_pivot_array = np.array(pivot_collection).T
    point_pivot_array = np.squeeze(point_pivot_array)
    return point_pivot_array


def angle(TDI_MULTI_TIMEFRAMES,timeframes,rsi_period,angle_max,angle_min):
    dataframe = []
    all_indices = []
    for i in range(len(TDI_MULTI_TIMEFRAMES)):
        data = TDI_MULTI_TIMEFRAMES[i]
        fast = data[f'fast_ma_{timeframes[i]}']
        low = data[f'low_ma_{timeframes[i]}']
        midle = data[f'midle_ma_{timeframes[i]}']
        fast_shifted = fast.shift(1)  # Décale les valeurs de "fast" d'une ligne vers le bas
        low_shifted = low.shift(1)    # Décale les valeurs de "low" d'une ligne vers le bas
        midle_shifted = midle.shift(1)  # Décale les valeurs de "midle" d'une ligne vers le bas
        angle_fast = np.arctan(fast - fast_shifted) * 180 / np.pi
        angle_low = np.arctan(low - low_shifted) * 180 / np.pi
        angle = (angle_low + (angle_fast * rsi_period / midle))/(1+rsi_period / midle)
        condition_angle = (angle<angle_max)&(angle>angle_min)
        print(f'nbr de condition angle respecter sur UT={timeframes[i]} ==> {condition_angle.sum()}')
        temp_df = pd.DataFrame({    ## Créer un dataframe temporaire avec les valeurs TDI et datetime en plus
                f'angle_{timeframes[i]}': condition_angle,
                f'cross_bull_{timeframes[i]}': TDI_MULTI_TIMEFRAMES[i][f'cross_bull_{timeframes[i]}'],
                f'trend_bull_{timeframes[i]}': TDI_MULTI_TIMEFRAMES[i][f'trend_bull_{timeframes[i]}'],
                f'cross_bear_{timeframes[i]}': TDI_MULTI_TIMEFRAMES[i][f'cross_bear_{timeframes[i]}'],
                f'trend_bear_{timeframes[i]}': TDI_MULTI_TIMEFRAMES[i][f'trend_bear_{timeframes[i]}'],
                'Datetime': data.index
            })

        temp_df.set_index('Datetime', inplace=True)   #Datetime en index
        temp_df.sort_index(inplace=True)     #Trier l'index en ordre croissant
        all_indices.append(temp_df.index)    #Dans all indices on rajoute seulement datetime
        dataframe.append(temp_df)
    reference_index = pd.Index(sorted(set.union(*map(set, all_indices))))  
    aligned_dfs = [
        df.reindex(reference_index, method='ffill') for df in dataframe
    ]
    combined_df = pd.concat(aligned_dfs, axis=1)   # Le TDI multi timeframe dans un dataframe
    return combined_df



#/////////////////////////////////////
# Fonction signal 
#/////////////////////////////////////


def entries(data,timeframes):
    conditions_per_timeframe_long = []
    conditions_per_timeframe_short = []
    #Faire une boucle For qui parcour timeframes.. pour chaque boucle tu fais un ET logique 
    for i in timeframes:
        entries_long = (data[f'cross_bull_{i}'] & data[f'trend_bull_{i}'] & data[f'angle_{i}'])
        entries_short = (data[f'cross_bear_{i}'] & data[f'trend_bear_{i}'] & data[f'angle_{i}'])
        
       
        conditions_per_timeframe_long.append(entries_long)
        conditions_per_timeframe_short.append(entries_short)


    
    final_condition_long = conditions_per_timeframe_long[0]
    final_condition_short = conditions_per_timeframe_short[0]
    for condition in conditions_per_timeframe_long[1:]:
        final_condition_long &= condition
        
    for condition in conditions_per_timeframe_short[1:]:
        final_condition_short &= condition

    print(final_condition_short.shape)
    print(final_condition_long.shape)
    return final_condition_long,final_condition_short

def exit(entries_long, entries_short, close, point_pivot, pivot_tp_sl):
    # Initialiser les séries de sorties avec False
    exits_long = pd.Series(False, index=entries_long.index)
    exits_short = pd.Series(False, index=entries_short.index)

    # Gestion des sorties longues
    for i in range(len(entries_long)):
        if entries_long.iloc[i]:  # Si une entrée longue est active
            tp = point_pivot[f'pivot_resistance_{pivot_tp_sl}'].iloc[i]
            sl = point_pivot[f'pivot_support_{pivot_tp_sl}'].iloc[i]

            # Vérifier que les indices sont valides avant d'accéder à close
            for j in range(i + 1, len(close)):
                if j >= len(close):
                    break  # Éviter l'erreur out-of-bounds
                if close.iloc[j] >= tp:  # Si le prix atteint le take profit
                    exits_long.iloc[j] = True
                    break
                if close.iloc[j] <= sl:  # Si le prix atteint le stop loss
                    exits_long.iloc[j] = True
                    break

    # Gestion des sorties courtes
    for i in range(len(entries_short)):
        if entries_short.iloc[i]:  # Si une entrée courte est active
            tp = point_pivot[f'pivot_support_{pivot_tp_sl}'].iloc[i]
            sl = point_pivot[f'pivot_resistance_{pivot_tp_sl}'].iloc[i]

            # Vérifier que les indices sont valides avant d'accéder à close
            for j in range(i + 1, len(close)):
                if j >= len(close):
                    break  # Éviter l'erreur out-of-bounds
                if close.iloc[j] <= tp:  # Si le prix atteint le take profit
                    exits_short.iloc[j] = True
                    break
                if close.iloc[j] >= sl:  # Si le prix atteint le stop loss
                    exits_short.iloc[j] = True
                    break

    # Retourner les résultats
    print(exits_long.shape)
    print(exits_short.shape)
    return exits_long, exits_short


    
    



#////////////////////////////////
#Custom Indicator vectorbt 
#////////////////////////////////



Ind_TDI = vbt.IndicatorFactory(
    class_name="tdi",
    input_names= ["rsi"],
    param_names = ["fast_period","low_period","midle_period"],
    output_names= ['fast_ma','low_ma','midle_ma']
).from_apply_func(TDI)



#////////////////////////////////
#Stratégie
#////////////////////////////////




class Strategie:

    def __init__(self,data,tickers,timeframes):
        self.data = data
        self.tickers = tickers
        self.timeframes = timeframes
    

    def backtest(self,fast_period_tdi,low_period_tdi,midle_period_tdi,params_entry):
        TDI = self.make_value(fast_period_tdi,low_period_tdi,midle_period_tdi)
        print(TDI)


    def make_value(self, fast_period_tdi, low_period_tdi, midle_period_tdi):
        """
        Calcule les indicateurs TDI pour différentes périodes de temps,
        et combine les résultats dans un DataFrame aligné sur un index de référence qui est la date et l'heure
        """
        TDI_MULTI_TIMEFRAME = []
        all_indices = []
        for i in self.timeframes:
            close = self.data[f'USDJPY_{i}']['close']
            rsi = vbt.RSI.run(close, window=24)
            TDI = Ind_TDI.run(rsi.rsi, fast_period_tdi, low_period_tdi, midle_period_tdi)
            displace_fast_ma = TDI.fast_ma.shift(2)
            displace_low_ma = TDI.low_ma.shift(2)
            displace_midle_ma = TDI.midle_ma.shift(1)
            cross_bull = (TDI.fast_ma > TDI.low_ma) & (displace_fast_ma < displace_low_ma)
            cross_bear = (TDI.fast_ma < TDI.low_ma) & (displace_fast_ma > displace_low_ma)
            print(f'nbr de cross bear sur UT={i} est nbr_cross_bear={cross_bear.sum()}')
            print(f'nbr de cross bull sur UT={i} est nbr_cross_bull={cross_bull.sum()}')
            trend_bull = displace_midle_ma < TDI.midle_ma
            trend_bear = displace_midle_ma > TDI.midle_ma
            temp_df = pd.DataFrame({    ## Créer un dataframe temporaire avec les valeurs TDI et datetime en plus
                f'fast_ma_{i}': TDI.fast_ma,
                f'low_ma_{i}': TDI.low_ma,
                f'midle_ma_{i}': TDI.midle_ma,
                f'cross_bull_{i}': cross_bull,
                f'cross_bear_{i}': cross_bear,
                f'trend_bull_{i}': trend_bull,
                f'trend_bear_{i}': trend_bear,
                'Datetime': self.data[f'USDJPY_{i}']['Datetime']
            })
            temp_df.set_index('Datetime', inplace=True)   #Datetime en index
            temp_df.sort_index(inplace=True)     #Trier l'index en ordre croissant
            all_indices.append(temp_df.index)    #Dans all indices on rajoute seulement datetime
            TDI_MULTI_TIMEFRAME.append(temp_df)  #Ici on rajoute les valeurs


        angle_time_frame = angle(TDI_MULTI_TIMEFRAME,self.timeframes,24,80,20)
        data = self.data[f'USDJPY_W1']
        point_pivot_value = point_pivot(data['close'],data['high'],data['low'],4) # Point pivot dans array 
        columns = []
        count = 0
        for i in range(point_pivot_value.shape[1]):
            if i % 2 == 0:
                columns.append(f"pivot_support_{count}") 
            else:
                columns.append(f"pivot_resistance_{count}") 
                count += 1
        point_pivot_df = pd.DataFrame(point_pivot_value,columns=columns,index=data['Datetime'])
        point_pivot_df = point_pivot_df.reindex(angle_time_frame.index, method='ffill')

        # Concaténer les deux DataFrames sur l'index
        # pivot_and_TDI = pd.concat([angle_time_frame, point_pivot_df], axis=1)  # Ici j'ai bien le dataframe avec toute les colones de toute les valeurs que j'ai besoin 
        point_pivot_df = point_pivot_df.round(1)
        print(point_pivot_df)
        entries_long,entries_short = entries(angle_time_frame,['H4','H1','M30','M15'])
        exits_long, exits_short = exit(entries_long,entries_short,self.data[f'USDJPY_M15']['close'],point_pivot_df,2)
        close = self.data[f'USDJPY_M15']['close']
        target_length = 522861

        # Vérifier si un ajustement est nécessaire
        if len(close) < target_length:
            # Créer une série avec des zéros pour compléter la différence
            additional_zeros = pd.Series(1, index=range(len(close), target_length))
            # Étendre la série close
            close = pd.concat([close, additional_zeros])
        print(close.shape)
        portfolio = vbt.Portfolio.from_signals(
            close,
            entries_long,
            exits_long,
            entries_short,
            exits_short,
            init_cash=1000,  # Capital initial
            fees=0.001,  # Frais de transaction (0.1%)
        )

        print(portfolio.total_return())



    def make_entries(dataframe,angle_min,angle_max):
        pass

       


RSI_PERIOD = 24
TDI_FAST_PERIOD = 2
TDI_LOW_PERIOD = 7
TDI_MIDDLE_PERIOD = 34
TDI_ANGLE_MIN = 20
TDI_ANGLE_MAX = 80

tickers = ['USDJPY']
timeframes= ['D1','H4','H1','M30','M15']
strat = Strategie(USDJPY,tickers,timeframes)
strat.backtest(TDI_FAST_PERIOD,TDI_LOW_PERIOD,TDI_MIDDLE_PERIOD,'4')


"""
MAJ 14 janvier
Tu as fini le dataframe avec le calcul des indicateurs 

-> Tu dois faire la fonction entrée et la fonction sortie 

-> Faire le backtest -> Sur 2015-2025 histoire d'avoir au moins une centaine de trade

-> Partie optimisation.. ? ca va etre important de faire le WFA et le profile de parametre mais va avoir beaucoup de probleme..
    - Les parametres a opti -> fast et low period (comme une paire)
    - les TP / SL 

    Je vais avoir des soucis je pense avec la partie optimisation et variation des variables -> A voir si je le fais ou pas.. pas envie de perdre 5 jours

-> Monte Carlo test

-> Si rentable faire simulation propfirm... 

-> Appliquer la strat selon ce qui est le plus rentable comme approche 

"""





"""

A noter pour la fin de semaine et le travail du dimanche :
-> Mon code python n'est en rien applicable a du réel.. il faudrat  tout recoder si tu veux appliquer la strat en réel.. 
-> Le multi timeframe c'est l'usine j'ai beaucoup de mal a faire un truc propre est compréhensible
    - De ce point la il en découle plein de truc comme le manque de skils avec panda et le changement d'index etc 


"""