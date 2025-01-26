import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

INDICE_PATH = r'C:\Users\Jordi\Desktop\Environement de developement\Data\IndiceHub\{0}.csv'
INDICE_TICK_PATH = r'C:\Users\Jordi\Desktop\Environement de developement\Data\IndiceHubTicks\{0}.csv'
FOREX_PATH = r'C:\Users\Jordi\Desktop\Environement de developement\Data\ForexHub\{0}.csv'

#-------------------------------------------------DATA GESTION-------------------------------------------------#


def get_data_indice(tickers, frequence, start_period=None, end_period=None):
    data_dict = {}  
    max_length = 0  

    for ticker in tickers:
        file_path = INDICE_PATH.format(ticker)
        try:
            # Chargement et uniformisation
            df = pd.read_csv(file_path)
            df = uniform_convention_data(df)  # Datetime est maintenant l'index

            # Filtrage via l'index
            if start_period:
                start_date = pd.to_datetime(start_period)
                df = df.loc[df.index >= start_date]
            if end_period:
                end_date = pd.to_datetime(end_period)
                df = df.loc[df.index <= end_date]
            # Resample
            df = resample_and_return(df, frequence)
            # Mettre à jour la longueur maximale
            max_length = max(max_length, len(df))
            data_dict[ticker] = df

        except Exception as e:
            print(f"Erreur lors du chargement des données pour {ticker}: {e}")

    # Uniformisation de la taille (version corrigée)
    for ticker, df in data_dict.items():
        if len(df) < max_length:
            fill_data = {col: [np.nan]*(max_length - len(df)) for col in df.columns}
            data_dict[ticker] = pd.concat([df, pd.DataFrame(fill_data)], ignore_index=False)

    return data_dict




def get_data_forex(tickers, frequence, start_period=None, end_period=None):
    data_dict = {}  
    max_length = 0  

    for ticker in tickers:
        file_path = FOREX_PATH.format(ticker)
        try:
            # Chargement et uniformisation
            df = pd.read_csv(file_path)
            df = uniform_convention_data(df)  # Datetime est maintenant l'index

            # Filtrage via l'index
            if start_period:
                start_date = pd.to_datetime(start_period)
                df = df.loc[df.index >= start_date]
            if end_period:
                end_date = pd.to_datetime(end_period)
                df = df.loc[df.index <= end_date]
            # Resample
            df = resample_and_return(df, frequence)
            # Mettre à jour la longueur maximale
            max_length = max(max_length, len(df))
            data_dict[ticker] = df

        except Exception as e:
            print(f"Erreur lors du chargement des données pour {ticker}: {e}")

    # Uniformisation de la taille (version corrigée)
    for ticker, df in data_dict.items():
        if len(df) < max_length:
            fill_data = {col: [np.nan]*(max_length - len(df)) for col in df.columns}
            data_dict[ticker] = pd.concat([df, pd.DataFrame(fill_data)], ignore_index=False)

    return data_dict


def get_data_indice_tick(tickers,tick,start_period=None,end_period=None):
    data_dict = {}  
    max_length = 0  

    for ticker in tickers:
        file_path = INDICE_TICK_PATH.format(ticker)
        try:
            # Chargement et uniformisation
            df = pd.read_csv(file_path)
            df = aggregate_ticks_to_bars(df,tick)
            df = uniform_convention_data(df)  # Datetime est maintenant l'index

            # Filtrage via l'index
            if start_period:
                start_date = pd.to_datetime(start_period)
                df = df.loc[df.index >= start_date]
            if end_period:
                end_date = pd.to_datetime(end_period)
                df = df.loc[df.index <= end_date]
            # Resample
            
            # Mettre à jour la longueur maximale
            max_length = max(max_length, len(df))
            data_dict[ticker] = df

        except Exception as e:
            print(f"Erreur lors du chargement des données pour {ticker}: {e}")

    # Uniformisation de la taille (version corrigée)
    for ticker, df in data_dict.items():
        if len(df) < max_length:
            fill_data = {col: [np.nan]*(max_length - len(df)) for col in df.columns}
            data_dict[ticker] = pd.concat([df, pd.DataFrame(fill_data)], ignore_index=False)

    return data_dict

def get_data_crypto():
    pass



def uniform_convention_data(df):
    # Création/formatage de la colonne Datetime
    if 'Datetime' not in df.columns:
        # Combine Date et Time en vérifiant le format
        try:
            # Vérifie la présence des colonnes
            if 'Date' not in df.columns or 'time' not in df.columns:
                raise KeyError("Colonnes 'Date' ou 'time' manquantes")
            
            # Combine en forçant le format HH:MM:SS
            datetime_str = (
                df['Date'].astype(str) + ' ' + 
                df['time'].astype(str).str.split(':').str[:2].str.join(':') + ':00'
            )
            
            # Conversion stricte
            df['Datetime'] = pd.to_datetime(
                datetime_str,
                format='%Y-%m-%d %H:%M:%S',
                errors='coerce'
            )
            
        except Exception as e:
            raise ValueError(f"Erreur de conversion : {str(e)}")
    else:
        # Formatage existant si nécessaire
        df['Datetime'] = pd.to_datetime(
            df['Datetime'],
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce'
        )
    
    # Nettoyage des données invalides
    invalid_count = df['Datetime'].isna().sum()
    if invalid_count > 0:
        print(f"⚠️ {invalid_count} lignes avec format de date invalide supprimées")
        df = df.dropna(subset=['Datetime'])
    
    # Suppression des colonnes originales
    df = df.drop(columns=['Date', 'time'], errors='ignore')
    
    # Réindexation finale
    df = df.set_index('Datetime').sort_index()
    
    return df
    
    


# Print la data + renvoie un print avec écris si la Data est bonne ou non. 
# Il faiut plusieurs test pour voir si y'a pas des trou dans la data ou des valeurs abberante 


def is_weekday(date):
    return date.weekday() < 5


def resample_and_return(df,timeframes):
    timeframe_to_freq = {
        'M1': '1T',   # 1 minute
        'M5': '5T',   # 5 minutes
        'M15': '15T', # 15 minutes
        'M30': '30T', # 30 minutes
        'H1': '1H',   # 1 heure
        'H4': '4H',   # 4 heures
        'D1': '1D',    # 1 jour
        'W1': '1W'  #1 week
    }
    if timeframes not in timeframe_to_freq:
        print(f"timeframe '{timeframes}' non pris en charge. Ignoré.")
        return 0
    else:
        freq = timeframe_to_freq[timeframes]
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
        return ohlc_resampled
    

def aggregate_ticks_to_bars(df, ticks_per_bar):
    df_final = pd.DataFrame()

    df['tick_index'] = df.index // ticks_per_bar
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['is_weekday'] = df['Date'].apply(is_weekday)
    data_weekdays = df[df['is_weekday']].copy() 

    data_weekdays.loc[:, 'buy_volume'] = data_weekdays['size'].where(data_weekdays['side'] == 'B', 0)  # Achat market (side = A)
    data_weekdays.loc[:, 'sell_volume'] = data_weekdays['size'].where(data_weekdays['side'] == 'A', 0)  # Vente market (side = B)


    ohlc_dict = {
        'price': ['first', 'max', 'min', 'last'],
        'size': 'sum',
    }
    aggregated_data = data_weekdays.groupby('tick_index').agg(ohlc_dict)
    
    aggregated_data.columns = ['open', 'high', 'low', 'close', 'volume']

    aggregated_data['delta'] = data_weekdays.groupby('tick_index')['buy_volume'].sum() - data_weekdays.groupby('tick_index')['sell_volume'].sum()
    aggregated_data['CVD'] = aggregated_data['delta'].cumsum()

    aggregated_data['Date'] = data_weekdays.groupby('tick_index')['Date'].first().values
    aggregated_data['Time'] = data_weekdays.groupby('tick_index')['Time'].first().values

    df_final = pd.concat([df_final, aggregated_data])

    return df_final
    # df_final.to_csv(f'aggregated_2023_{ticks_per_bar}_ticks.csv', index=False)

#------------------------------------------------------------------------------------------------------------------#







#---------------------------------------------------PRINT ELEMENT--------------------------------------------------#

"""
Precondition : Un dataframe OHLC,  un dataframe avec au moins un side c'est a dire entries et short
Postcondition : print un graphique movible
"""
def print_trades(entries_long,entries_short,exits_long,exits_short,data):
    if 'Datetime' not in data.columns:
        data['Datetime'] = data['Date']+ ' ' + data['time']

    fig = go.Figure(data=[go.Scatter(
        x=data['Datetime'],
        y=data['close'],
        mode='lines',
        name='Prix de clôture',
        line=dict(color='#00ff88', width=1)  # Couleur personnalisable
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


def get_ohlc_print(data):
    fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['open'],
    high=data['high'],
    low=data['low'],
    close=data['close'],
    increasing_line_color='#2ECC71',  # Couleur des bougies haussières
    decreasing_line_color='#E74C3C'   # Couleur des bougies baissières
    )])
    # Personnalisation supplémentaire
    fig.update_layout(
        title="Graphique OHLC",
        xaxis_title="Date/Heure",
        yaxis_title="Prix",
        template="plotly_dark",
        xaxis_rangeslider_visible=False
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

#-----------------------------------------------------------------------------------------------------------------#

