import numpy as np
import pandas as pd 
import vectorbt as vbt
vbt.settings.plotting["layout"]["template"] = "vbt_dark"
import matplotlib.pyplot as plt
plt.style.use("dark_background")
from datetime import datetime
import MetaTrader5 as mt5



MAJOR_PATH = r'C:\Users\Jordi\Desktop\Environement de developement\Trading_Dev_Stratégie_Environement\Data\DataHub/{0}.csv'



def get_data_forex(symbols, timeframe=mt5.TIMEFRAME_M1):
    """
    Utilise metatrader 5
    Récupère les données Forex pour les symboles donnés et les retourne sous forme de dictionnaire de DataFrames.

    :param symbols: Liste des symboles à importer (ex: ["EURUSD", "GBPUSD"]).
    :param timeframe: Période des chandeliers (par défaut 1 minute : mt5.TIMEFRAME_M1).
    :param num_candles: Nombre de chandeliers à récupérer (par défaut 100).
    :return: Dictionnaire contenant les DataFrames pour chaque symbole avec colonnes :
             [date, time, symbol, open, high, low, close, volume, swap_long, swap_short].
    """
    if not mt5.initialize():
        print("Erreur d'initialisation de MetaTrader5")
        return None
    

    data_dict = {}
    start_date = datetime(2006, 1, 1)
    for symbol in symbols:
        # Vérifier si le symbole est disponible
        info = mt5.symbol_info(symbol)
        if info is None:
            print(f"Le symbole {symbol} n'est pas disponible.")
            continue

        # Activer le symbole si nécessaire
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                print(f"Impossible de sélectionner le symbole {symbol}.")
                continue

        # Récupérer les données historiques
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, datetime.now())
        if rates is None:
            print(f"Erreur lors de la récupération des données pour {symbol}.")
            continue

        # Transformer les données en DataFrame
        df = pd.DataFrame(rates)
        df['date'] = pd.to_datetime(df['time'], unit='s').dt.date
        df['time'] = pd.to_datetime(df['time'], unit='s').dt.time
        df['symbol'] = symbol

        # Ajouter les informations sur les swaps
        df['swap_long'] = info.swap_long
        df['swap_short'] = info.swap_short

        # Sélectionner et réorganiser les colonnes
        df = df[['date', 'time', 'symbol', 'open', 'high', 'low', 'close', 'tick_volume', 'swap_long', 'swap_short']]
        df.rename(columns={'tick_volume': 'volume'}, inplace=True)

        # Ajouter le DataFrame au dictionnaire
        data_dict[symbol] = df

    return data_dict
     





def get_data(list_tickers, start=None, end=None):
    data_dict = {}  

    max_length = 0  

    for ticker in list_tickers:
        file_path = MAJOR_PATH.format(ticker)
        try:
            # Charger le fichier CSV directement dans un DataFrame
            df = pd.read_csv(file_path)

            # Si une colonne 'Date' existe, appliquer le filtrage
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                if start:
                    df = df[df['Date'] >= start]
                if end:
                    df = df[df['Date'] <= end]

            # Mettre à jour la longueur maximale
            max_length = max(max_length, len(df))

            # Ajouter le DataFrame au dictionnaire
            data_dict[ticker] = df
        except Exception as e:
            print(f"Erreur lors du chargement des données pour {ticker}: {e}")

    # Uniformiser la taille des colonnes avec une valeur par défaut (par exemple, 0)
    for ticker, df in data_dict.items():
        if len(df) < max_length:
            additional_rows = pd.DataFrame(
                {col: [0] * (max_length - len(df)) for col in df.columns}
            )
            data_dict[ticker] = pd.concat([df, additional_rows], ignore_index=True)

    return data_dict


         



def cut_data_by_dates(df, date_col, start_date, end_date):
    """
    Filtre un DataFrame en fonction d'une plage de dates.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        date_col (str): Le nom de la colonne contenant les dates au format "YYYY-MM-DD".
        start_date (str): La date de début au format "YYYY-MM-DD".
        end_date (str): La date de fin au format "YYYY-MM-DD".

    Returns:
        pd.DataFrame: Le DataFrame filtré entre les dates spécifiées.
    """
    df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce")
    start_date = pd.to_datetime(start_date, format="%Y-%m-%d", errors="coerce")
    end_date = pd.to_datetime(end_date, format="%Y-%m-%d", errors="coerce")
    if pd.isnull(start_date) or pd.isnull(end_date):
        raise ValueError("La date de début ou de fin est invalide.")
    filtered_df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)].copy()

    return filtered_df


"""
Objectif :
Utiliser dans la partie optimisation pour pouvoir retrouver le parametre optimal 


Précond :
Cette fonction prend en parametre 2 tableau non vide et non nulle
2 tableau avec des valeurs nuémrique

Poscond :
renvoie un tableau avec chaque combinaison de combinaison 1 * each(combinaison2)
"""
def get_combinaison(combinaison1,combinaison2):
    liste_combinaison=[]
    for i in combinaison1:
        for j in combinaison2:
            liste_combinaison.append((i,j))
    return liste_combinaison



#################################################
#Affichage pnl


"""
Objectif :
Afficher le PNL d'un portfolio sur navigateur

Précond :
prend en parametre un object vectorbt qui comprend le résultat d'un backtest


Poscond :
print un pnl

"""
def get_pnl(portfolio):
    equity = portfolio.value()
    equity.vbt.plot().show()


def get_mean_pnl_multi_asset(portfolio):
    all_values = portfolio.value()  
    mean_pnl = all_values.mean(axis=1) 
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(mean_pnl, label="Courbe PnL Moyenne (Lissée)")
    plt.xlabel("Temps")
    plt.ylabel("Valeur Moyenne du Portefeuille")
    plt.title("Performance Moyenne du Portefeuille Multi-Actifs")
    plt.legend()
    plt.show()



####################################################################
##Metrics 


def get_metrics(portfolio,params):
    if params == 1:
        return portfolio.total_return().mean()
    if params == 2:
        return portfolio.sharpe_ratio().mean()
    if params == 3:
        return portfolio.calmar_ratio().mean()
    if params == 4:
        return  portfolio.win_rate().mean()



## Fonction qui permet de récuperer la metrics de perf souhaité et de l'associer a ton parametre lors d'une opti
##Gère seulement les cas ou on a un actif a opti
def get_df_optimisation_parameter(combinaison,portfolio,metrics):
    value_metrics = 0
    all_metrics = []
    for i in range(len(combinaison)):
        value_metrics = get_metrics(portfolio[i],metrics)
        all_metrics.append(value_metrics)
    results_df = pd.DataFrame({
        'combinaison': combinaison,
        'metrics': all_metrics,
    })
    return results_df


"""
*** certif Multi Asset 

Objectif :
Une focntion qui permet de renvoyer un dataframe comprenant toute les informations voulu sur une optimisation de params


Précond:
Parametre de toute les combinaisons, et de la liste des potfolio 

Postcond :
renvoie un dataframe
"""
def get_information_combinaison_optimistion(dictionaire, combinaison):
    results = {} 
    for i in range(len(combinaison)):
        value_metrics = 0
        for df in dictionaire.values():
            value_metrics = df['metrics'].iloc[i] + value_metrics
        value_metrics = value_metrics / len(dictionaire)
        results[combinaison[i]] = value_metrics
    results_df = pd.DataFrame([results])
    return results_df


    

def get_info_limited_testing_Core_system(portfolio):
    nbr_trades = portfolio.trades.count()
    expectency = portfolio.trades.expectancy()
    win_rate = portfolio.trades.win_rate()
    total_return = portfolio.total_return()
    liste_value = [nbr_trades,expectency,win_rate,total_return]
    return liste_value




## Recup le Average Win et le Average loose et comparer.. si tu vois que ta un ptot win rate et un ratio win/loose faible -> mettre défavorable 
## Cette fonction est fairep pour prendre un lot de portfolio et voir la parité favorable / défavorable
def parite_limited_testing_core(all_portfolio):
    count_win_rate = 0
    count_total_return = 0
    nbr_trades = 0
    expectency = 0
    win_rate = 0
    total_return = 0

    for i in range(len(all_portfolio)):
        print(all_portfolio['win_rate'].iloc[i])
        if all_portfolio['win_rate'].iloc[i] > 0.52:
            count_win_rate += 1
        if all_portfolio['total_return'].iloc[i] > 0.05:
            count_total_return += 1

        nbr_trades += all_portfolio['nb_trades'].iloc[i]
        expectency += all_portfolio['expectancy'].iloc[i]
        win_rate += all_portfolio['win_rate'].iloc[i]
        total_return += all_portfolio['total_return'].iloc[i]
    print(count_win_rate)
    win_rate = win_rate / len(all_portfolio)
    expectency = expectency / len(all_portfolio)

    print("------------------RECAP---------------------")
    print('\n')
    print(f"Trade en moyenne par backtest = {nbr_trades/len(all_portfolio)}")
    print(f"Expectency moyenne par backtest = {expectency}")
    print(f"Win rate en moyenne par backtest = {win_rate}")
    print(f"Total return en moyenne par backtest = {total_return/len(all_portfolio)}")

    ratio = count_win_rate / len(all_portfolio)
    complementary_ratio = 1 - ratio
    labels = ['Ratio gagnant', 'Ratio perdant']  # Les labels pour chaque partie
    sizes = [ratio, complementary_ratio]  # La taille des portions
    colors = ['green', 'red']  # Les couleurs pour chaque portion
    explode = (0.1, 0)  # Exploser la première portion pour la mettre en avant (facultatif)

    # Création du camembert
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

    # Ajouter un titre
    plt.title('Répartition du ratio des trades gagnants et perdants')

    # Afficher le graphique
    plt.axis('equal')  # Pour que le graphique soit circulaire
    plt.show()


        

#---------------------------------------------------------------------------------------------------------------------------------------#

#Fonction SPLIT
"""

Fcontion Split pour le WFA
Attention cette fonction prend en compte seulement le découpage en année donc faire 1 an et demis pour OOS c'est pas possible par exemple 

"""

def split(list_tickers,period_start,period_end,period_OOS,period_IOS):
    cycles = []
    current_start = period_start

     # Convertir les dates en années sous forme d'entiers
    start_year = datetime.strptime(period_start, "%Y-%m-%d").year
    end_year = datetime.strptime(period_end, "%Y-%m-%d").year

    cycles = []
    current_start = start_year

    while current_start + period_IOS + period_OOS <= end_year:
        # Définir les périodes IOS
        cycle_IOS_start_year = current_start
        cycle_IOS_end_year = current_start + period_IOS - 1

        # Définir les périodes OOS
        cycle_OOS_start_year = cycle_IOS_end_year + 1
        cycle_OOS_end_year = cycle_OOS_start_year + period_OOS - 1

        # Formater les périodes en chaînes de caractères
        cycle_IOS_start = f"{cycle_IOS_start_year}_01_01"
        cycle_IOS_end = f"{cycle_IOS_end_year}_12_31"
        cycle_OOS_start = f"{cycle_OOS_start_year}_01_01"
        cycle_OOS_end = f"{cycle_OOS_end_year}_12_31"

        # Ajouter le cycle à la liste
        cycles.append({
            "IOS": (cycle_IOS_start, cycle_IOS_end),
            "OOS": (cycle_OOS_start, cycle_OOS_end)
        })

        # Passer au cycle suivant
        current_start += 1

    result= {}
    for ticker in list_tickers:
        result[ticker] = {}
        
        for cycle_index, n in enumerate(cycles, start=1):
            cycle_name = f"cycle_{cycle_index}"
            
            # Convertir les dates au bon format pour éviter les erreurs
            IOS_date_start = datetime.strptime(n['IOS'][0], "%Y_%m_%d")
            IOS_date_end = datetime.strptime(n['IOS'][1], "%Y_%m_%d")
            OOS_date_start = datetime.strptime(n['OOS'][0], "%Y_%m_%d")
            OOS_date_end = datetime.strptime(n['OOS'][1], "%Y_%m_%d")
            
            # Appeler get_data avec les dates au format datetime
            IOS_data = get_data([ticker], IOS_date_start, IOS_date_end)
            OOS_data = get_data([ticker], OOS_date_start, OOS_date_end)

            # Enregistrer les données dans le dictionnaire
            result[ticker][cycle_name] = {
                "IOS": IOS_data,
                "OOS": OOS_data
            }

    return result










"""

Info utile 
Mean DD => portfolio.Drawdonws.avg_drawdown
cagr = portfolio.annualized_return()
max_dd = portfolio.max_drawdown()

"""

# symbols = ["EURUSD"]
# data = get_data_forex(symbols, timeframe=mt5.TIMEFRAME_H1)
# eurusd_data = data["EURUSD"]
# plt.figure(figsize=(12, 6))
# plt.plot(eurusd_data['date'], eurusd_data['close'], label="Clôture EURUSD")
# plt.title("Clôture EURUSD")
# plt.xlabel("Date")
# plt.ylabel("Prix de clôture")
# plt.legend()
# plt.grid()
# plt.show()
