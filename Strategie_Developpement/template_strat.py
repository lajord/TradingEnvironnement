import numpy as np
import pandas as pd 
import vectorbt as vbt
import talib 
import matplotlib.pyplot as plt 

import sys
import os
chemin_parent = os.path.abspath(r'C:\Users\Jordi\Desktop\Trading_Dev_Stratégie_Environement\FunctionEssential')

if chemin_parent not in sys.path:
    sys.path.append(chemin_parent)

import Trading_Dev_Stratégie_Environement.FunctionEssential.function_essential as fe


#################################################################################################################

"""
Création des fonctions python pour les indicateurs 
=> MA ,RSI, ATR etc etc calculer avec les outils a disposition ! vectorbt
"""






###################################################################################################################


"""
Création des fonctions d'entrée et de sortie 
"""




####################################################################################################################
"""
Création des Indicator Factory
"""




class Strategie():


    def __init__(self,data,frequence,list_tickers):
        self.data = data
        self.frequence = frequence
        self.list_tickers = list_tickers


    def backtest(self,period_start,period_end, period1 , period2, params1 ):
        if (period_end == 0) and (period_start == 0):

            entries_df = pd.DataFrame()
            exits_df = pd.DataFrame()
            close_df = pd.DataFrame()

            for ticker in self.list_tickers:
                """
                A modifier pour rajouter calcul d'indicateur un peu spécial ou autre 
                """
                liste_value = self.make_value_no_combi()
                entries = self.make_entries()
                exits = self.make_exits()


                #Reset les index pour tout normaliser
                entries = entries.reset_index(drop=True)
                exits = exits.reset_index(drop=True)
                close = close.reset_index(drop=True)
                #enregistrer dans des dataframes 
                entries_df[ticker] = entries
                exits_df[ticker] = exits
                close_df[ticker] = close

            #remplacer les 0 qui ont état mis durant import data
            close_df = close_df.replace(0, None).fillna(method='ffill')
            portfolio = vbt.Portfolio.from_signals(
                close_df,
                entries_df,
                exits_df,
                freq=self.frequence
            )

            return portfolio
        
        else :
            pass


    def optimize(self,metrics,choice,period1,period2,params1):
        dataframe_save = {}   # Dans le cadre d'un multi Asset ca va venir enregistrer le df de l'optimisation -> Puis faire un mean -> Voir le parametre qui fonctionne le mieux sur tous
        if choice == 1:
            entries_df = pd.DataFrame()
            exits_df = pd.DataFrame()
            for ticker in self.list_tickers:

                entries_df = entries_df[0:0]
                exits_df = exits_df[0:0]
                data = self.data[ticker]
                liste_value = self.make_value_no_combi()

                for i in range(params1.shape[1]):  ## Ici tu boucles sur le paramètre que tu veux opti / Il peut y avoir deux boucles si tu opti deux params !
                    params1_1D = params1.iloc[:, i]
                    liste_value[0] =  params1_1D
                    entries = self.make_entries()
                    exits = self.make_exits()

                    #Important a pas toucher
                    entries_df[i] = entries 
                    exits_df[i] = exits
                    #-----------------------#

                # Important a note !! Pour pas avoir d'erreur ensuite les noms des colonnes de entries et exits doivent avoir des index de 0 a n (0,1,2,3 etc)
                # Si tu as un soucis avec la suite apres le backtest regarde les indices de entries et exits

                close = data['close']
                close_df = close.replace(0, None).fillna(method='ffill')
                portfolio = vbt.Portfolio.from_signals(
                     close_df,
                     entries_df,
                     exits_df,
                     freq= self.frequence,
                     slippage=0.0005
                )
                combinaison = fe.get_combinaison(period1,period2)
                df = fe.get_df_optimisation_parameter(combinaison,portfolio,metrics)
                dataframe_save[ticker] = df



        if choice == 2:

            pass

        """
        on peut rajouter d'autre choice biensur
        """
    
    def make_entries(self,close,list_value,params1,period1):
        """
        A Custom 
        """
        pass

    def make_exits(self,close,entries,list_value,params2,period2):
        """
        A Custom 
        """

        pass
    
    def make_value_no_combi(self,high,low,close,period1,period2,period3,period4):
        """
        A Custom 
        """
        pass