import MetaTrader5 as mt5
import datetime
import pandas as pd 


"""
Objectif :

Pouvoir faire du multiactif backtest avec un seul backtest sans faire de boucle et faire du multitime frame pour calculer indicateur

Renvoie une collection :

-> Gère correctement le multi time frame
-> Gère correctement l'appel a plusieurs tickers en même temps 

postcondition :


il faut que tout les dataframe est :
    -> Le meme nombres d'index 
    -> les noms des colonnes ont a la fin le nom de la time frame ex : "close_M15 | high_M15 ..."
"""


"""
Faire une fonction uniformiser pour rendre le backtest multi asset possible
-> Tu calculs t'es signaux sans prendre en compte la longueur de chaque actif et avant de backtest tu remplis avec des False a la fin 
"""

"""
=> Tu géres dans un premier temps le multi asset pour fournir une collection simple d'utilisation avec pourquoi pas un itérateur ?
=> Tu gères avant de backtest la concaténation de tout les signaux pour avoir des signaux de long / short / exits etx avec le meme nombres
d'index en remplissant par du False puis le close en remplissant par 0 ? ou autre car je sais que ca fait bug sinon 
"""



# start_date = datetime(2025, 1, 1)   -> important

def import_data_mt5(tickers, timeframe, start_date):
    if not mt5.initialize():
        print("Echec de l'initialisation de MetaTrader5")
        return None
    
    data_dict = {}

    for ticker in tickers:
        print(f"Importation des données pour {ticker}")
        
        symbol_info = mt5.symbol_info(ticker)
        if symbol_info is None:
            print(f"Titre non valide : {ticker}")
            continue

        if not symbol_info.visible:
            print(f"Titre {ticker} non visible, tentative d'activation...")
            if not mt5.symbol_select(ticker, True):
                print(f"Impossible d'activer le titre {ticker}")
                continue

        for tf in timeframe:    
            mt5_timeframe = getattr(mt5, f"timeFRAME_{tf}", None)
            if mt5_timeframe is None:
                print(f"timeframe non valide : {tf}")
                continue

            # Récupérer les données depuis la date de début
            rates = mt5.copy_rates_range(
                ticker, mt5_timeframe, start_date, datetime.datetime.now()
            )
            
            # Vérifier si les données ont été récupérées correctement
            if rates is None or len(rates) == 0:
                print(f"Échec de l'importation des données pour {ticker} en {tf}")
                continue



            rates_df = pd.DataFrame(rates)
            
        
            # Vérifier que 'time' est une colonne présente dans le DataFrame
            if 'time' not in rates_df.columns:
                print(f"Problème: 'time' colonne manquante pour {ticker} en {tf}")
                continue
            
            rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')

            # Renommer les colonnes pour inclure le timeframe
            rates_df = rates_df.rename(columns=lambda x: f"{x}_{tf}" if x != "time" else "time")
            key = f"{ticker}_{tf}"
            data_dict[key] = rates_df

    return data_dict


def resample_and_save(df, timeframes, output_folder):
    """
    Regroupe un DataFrame en fonction des timeframes spécifiées et enregistre chaque regroupement sous forme de fichier CSV.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données OHLC à resampler.
        timeframes (list): Une liste de timeframes sous la forme ['M15', 'H4'].
        output_folder (str): Le chemin du dossier où les fichiers CSV seront enregistrés.

    Returns:
        None
    """
    # Dictionnaire pour convertir les timeframes en fréquences pandas
    timeframe_to_freq = {
        'M1': '1T',   # 1 minute
        'M5': '5T',   # 5 minutes
        'M15': '15T', # 15 minutes
        'M30': '30T', # 30 minutes
        'H1': '1H',   # 1 heure
        'H4': '4H',   # 4 heures
        'D1': '1D',    # 1 jour
        'W1': '1W'
    }
    
    for tf in timeframes:
        if tf not in timeframe_to_freq:
            print(f"timeframe '{tf}' non pris en charge. Ignoré.")
            continue

        # Conversion de la timeframe en fréquence pandas
        freq = timeframe_to_freq[tf]

        # Regrouper les données par la fréquence spécifiée
        ohlc_resampled = df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Supprimer les lignes où il n'y a pas de données
        ohlc_resampled.dropna(inplace=True)

        # Générer le nom du fichier
        output_file = f"{output_folder}/ohlc_{tf}.csv"

        # Sauvegarder le fichier CSV
        ohlc_resampled.to_csv(output_file)
        print(f"timeframe {tf} enregistré sous : {output_file}")


input_file = r"H:\Desktop\Data\USDJPY_M1.csv"
output_folder = r"H:\Desktop\Environement_Trading_Developement"  # Dossier de sortie

# Charger les données
df = pd.read_csv(input_file)

# Convertir 'Datetime' et définir comme index


df['Datetime'] = pd.to_datetime(df['Datetime'])

# Définir 'Datetime' comme index
df.set_index('Datetime', inplace=True)


# Appeler la fonction avec les timeframes souhaitées
timeframes = ['H1']
resample_and_save(df, timeframes, output_folder)
def fix_time_format(input_file, output_file):
    """
    Corrige le format de la colonne 'time' en ajoutant ':00' pour compléter le format hh:mm:ss.
    Combine ensuite 'date' et 'time' en une seule colonne 'timestamp' et sauvegarde dans un nouveau fichier CSV.

    Args:
        input_file (str): Chemin du fichier CSV d'entrée.
        output_file (str): Chemin du fichier CSV corrigé en sortie.

    Returns:
        None
    """
    # Charger le fichier CSV
    df = pd.read_csv(input_file)

    # Ajouter ":00" à la colonne 'time' pour compléter le format hh:mm:ss
    df['time'] = df['time'].apply(lambda x: f"{x}:00" if len(x) == 5 else x)

    # Créer une colonne 'timestamp' en combinant 'date' et 'time'
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['time'])

    # Supprimer les colonnes 'date' et 'time' si nécessaire
    df.drop(columns=['Date', 'time'], inplace=True)

    # Réorganiser les colonnes pour placer 'timestamp' en premier
    df = df[['timestamp'] + [col for col in df.columns if col != 'timestamp']]

    # Sauvegarder dans un nouveau fichier CSV
    df.to_csv(output_file, index=False)
    print(f"Fichier corrigé sauvegardé sous : {output_file}")


# # Exemple d'utilisation
# input_file = r"H:\Desktop\Environement_Trading_Developement\USDJPY_M1.csv"
# output_file = r"H:\Desktop\Environement_Trading_Developement\USDJPY_M1_fixed.csv"

# fix_time_format(input_file, output_file)