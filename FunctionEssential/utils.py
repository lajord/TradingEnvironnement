import plotly.graph_objects as go
import plotly.io as pio  
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import vectorbt as vbt
from binance.client import Client
from datetime import datetime
pio.templates.default = "plotly_dark"



INDICE_PATH = r'C:\Users\Jordi\Desktop\Environement de developement\Data\IndiceHub\{0}.csv'
INDICE_TICK_PATH = r'C:\Users\Jordi\Desktop\Environement de developement\Data\IndiceHubTicks\{0}.csv'
FOREX_PATH = r'C:\Users\Jordi\Desktop\Environement de developement\Data\ForexHub\{0}.csv'

# INDICE_PATH = r'H:\Desktop\Data\{0}.csv'
# INDICE_TICK_PATH = r'H:\Desktop\Data\{0}.csv'
# FOREX_PATH = r'H:\Desktop\Data\{0}.csv'

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
            print('process séparation ticks')
            df = aggregate_ticks_to_bars(df,tick)
            print("process uniformisation")
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


def get_data_crypto(ticker,start_date,end_date,frequence):
    client = Client('','')

    klines = client.get_historical_klines(ticker[0],frequence,start_date,end_date)
    # Create a DataFrame from the klines
    df = pd.DataFrame(klines, columns=[
                                        'open_time','open', 'high', 'low', 'close',
                                            'volume','close_time', 'dv','num_trades',
                                            'taker_buy_vol','taker_buy_base_vol', 'ignore'
                                    ])

    # Set Close time to datetime Type and as index
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    df.set_index('close_time', inplace=True)
    df.drop(columns=['open_time','ignore'], inplace=True)

    # Round close time to normalize
    df.index = df.index.round('min')

    # Convert columns to float type
    df = df.astype(float)

    return df



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
        'H1': '1h',   # 1 heure
        'H4': '4h',   # 4 heures
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
    


"""
Cette fonction je dois la refaire car on a un csv en 100 ticks donc c'est plus une question de Bid ou ask etc juste tu aggreges la 
"""

def aggregate_ticks_to_bars(df, ticks_per_bar):
    # Vérifier que ticks_per_bar est divisible par 100
    nbr_ligne = ticks_per_bar / 100
    if not nbr_ligne.is_integer():
        print(f"La division en {ticks_per_bar} n'est pas possible\n")
        return None

    # Ajouter un index de tick pour chaque groupe de ticks_per_bar
    df['tick_index'] = df.index // nbr_ligne

    # Convertir la colonne 'Datetime' en datetime
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    # Définir les colonnes à agréger et les fonctions d'agrégation
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'delta': 'sum',
        'CVD': 'last',
        'Datetime': 'last'
    }

    # Agréger les données par tick_index
    aggregated_data = df.groupby('tick_index').agg(ohlc_dict)

    # Renommer les colonnes pour correspondre aux noms souhaités
    aggregated_data = aggregated_data.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
        'delta': 'delta',
        'CVD': 'CVD',
        'Datetime': 'Datetime'
    })

    return aggregated_data

#------------------------------------------------------------------------------------------------------------------#







#---------------------------------------------------PRINT ELEMENT--------------------------------------------------#

"""
Precondition : Un dataframe OHLC,  un dataframe avec au moins un side c'est a dire entries et short
Postcondition : print un graphique movible
"""
def print_trades(long_short, entries_long, entries_short, exits_long, exits_short, data):
    
    numeric_index = np.arange(1, len(data['close']) + 1)


    fig = go.Figure(data=[go.Scatter(
        x=numeric_index,  # Utilisation de l'index numérique
        y=data['close'],
        mode='lines',
        name='Prix de clôture',
        line=dict(color='#00ff88', width=1)
    )])
    
    # Ajouter les entrées longues
    if entries_long.any():
        fig.add_trace(go.Scatter(
            x=numeric_index[entries_long],
            y=data['close'][entries_long],
            mode='markers',
            marker=dict(symbol='x', size=10, color='blue'),
            name='Entrée Long'
        ))
    
    # Ajouter les entrées courtes (si elles existent)
    if entries_short is not None and entries_short.any():
        fig.add_trace(go.Scatter(
            x=numeric_index[entries_short],
            y=data['close'][entries_short],
            mode='markers',
            marker=dict(symbol='x', size=10, color='red'),
            name='Entrée Short'
        ))
    
    # Ajouter les sorties longues (si elles existent)
    if exits_long is not None and exits_long.any():
        fig.add_trace(go.Scatter(
            x=numeric_index[exits_long],
            y=data['close'][exits_long],
            mode='markers',
            marker=dict(symbol='triangle-up', size=10, color='blue'),
            name='Sortie Long'
        ))
    
    # Ajouter les sorties courtes (si elles existent)
    if exits_short is not None and exits_short.any():
        fig.add_trace(go.Scatter(
            x=numeric_index[exits_short],
            y=data['close'][exits_short],
            mode='markers',
            marker=dict(symbol='triangle-down', size=10, color='red'),
            name='Sortie Short'
        ))

    # Mise en forme du graphique
    fig.update_layout(
        title="Visualisation des Trades",
        xaxis_title="Index Numérique",
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


def print_close(data):
    index_num = np.arange(1, len(data) + 1)
    fig = go.Figure(data=[go.Scatter(
        x=index_num,  # Utilisation de l'index numérique
        y=data['close'],
        mode='lines',
        name='Prix de clôture',
        line=dict(color='#00ff88', width=1)
    )])
    fig.show()


def get_pnl(portfolio):
    equity = portfolio.value()
    fig = equity.vbt.plot()
    fig.update_layout(template="plotly_dark")  
    fig.show()


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



#------------------------------------------------------METRICS----------------------------------------------------#


def generate_portfolio_report(portfolio, close):
    """
    Génère un rapport de métriques à partir d'un objet Portfolio de vectorbt et affiche les résultats.
    
    :param portfolio: Objet Portfolio de vectorbt
    :param benchmark_returns: Séries temporelles des rendements du benchmark pour calculer le beta

    
    """
    benchmark_return, benchmark_sharp = get_metrics_buy_and_hold(close)

    print('\n-----------------------REPORT METRICS-----------------------\n\n')
    print(f"Total Return : {round(portfolio.total_return().iloc[-1] * 100, 2)}%")
    print(f'Benchmark return : {round(benchmark_return.iloc[-1]*100,2)}%')
    print(f"CAGR : {round(portfolio.annualized_return().iloc[-1]*100,2)}%")
    print(f"Volatility : {round(portfolio.annualized_volatility().iloc[-1]*100,2)}%")
    print(f"Sharpe Ratio : {round(portfolio.sharpe_ratio().iloc[-1],2)}")
    print(f'Benchmark sharpe ratio : {round(benchmark_sharp.iloc[-1],2)}')
    print(f"Max Drawdown: {round(portfolio.max_drawdown().iloc[-1]*100,2)}%")
    print(f"Calmar Ratio : {round(portfolio.calmar_ratio().iloc[-1],2)}")
    print(f"Beta : {round(portfolio.beta().iloc[-1],2)}")
    print('\n\n--------------- REPORT METRICS TRADES--------------------')
    print('\n')
    print(f'Total trades : {round(portfolio.trades.count().iloc[-1],0)}')
    print(f"Total long trades {portfolio.trades.long.count().iloc[-1]}")
    print(f"Total short trades {portfolio.trades.short.count().iloc[-1]}")
    print(f"Win rate : {portfolio.trades.closed.win_rate().iloc[-1]}%")
    print(f"Wining streak : {portfolio.trades.winning_streak.max().iloc[-1]}")
    print(f"Loosing streak : {portfolio.trades.losing_streak.max().iloc[-1]}")
    print(f"Average winning trade : {round(portfolio.trades.winning.returns.mean().iloc[-1],2)*100}%")
    print(f"Average losing trade : {round(portfolio.trades.losing.returns.mean().iloc[-1],2)*100}%")
    print(f"Profit factor : {round(portfolio.trades.profit_factor().iloc[-1],2)}")
    print(f"Expectancy : {round(portfolio.trades.expectancy().iloc[-1],2)}%")
    print('\n-----------------------------------------------------------\n')


def generate_portfolio_report_order_function(portfolio, close):
    benchmark_return, benchmark_sharp = get_metrics_buy_and_hold(close)

    print('\n-----------------------REPORT METRICS-----------------------\n\n')
    print(f"Total Return : {round(portfolio.total_return() * 100, 2)}%")
    print(f'Benchmark return : {round(benchmark_return*100,2)}%')
    print(f"CAGR : {round(portfolio.annualized_return()*100,2)}%")
    print(f"Volatility : {round(portfolio.annualized_volatility()*100,2)}%")
    print(f"Sharpe Ratio : {round(portfolio.sharpe_ratio(),2)}")
    print(f'Benchmark sharpe ratio : {round(benchmark_sharp,2)}')
    print(f"Max Drawdown: {round(portfolio.max_drawdown()*100,2)}%")
    print(f"Calmar Ratio : {round(portfolio.calmar_ratio(),2)}")
    print(f"Beta : {round(portfolio.beta(),2)}")
    print('\n\n--------------- REPORT METRICS TRADES--------------------')
    print('\n')
    print(f'Total trades : {round(portfolio.trades.count(),0)}')
    print(f"Total long trades {portfolio.trades.long.count()}")
    print(f"Total short trades {portfolio.trades.short.count()}")
    print(f"Win rate : {portfolio.trades.closed.win_rate()}%")
    print(f"Wining streak : {portfolio.trades.winning_streak.max()}")
    print(f"Loosing streak : {portfolio.trades.losing_streak.max()}")
    print(f"Average winning trade : {round(portfolio.trades.winning.returns.mean(),2)*100}%")
    print(f"Average losing trade : {round(portfolio.trades.losing.returns.mean(),2)*100}%")
    print(f"Profit factor : {round(portfolio.trades.profit_factor(),2)}")
    print(f"Expectancy : {round(portfolio.trades.expectancy(),2)}%")
    print('\n-----------------------------------------------------------\n')


def get_metrics_buy_and_hold(close):
    """
    Backtest une stratégie Buy & Hold et affiche les métriques.
    
    :param data: Données OHLC du benchmark
    :return: Rendement total du benchmark
    """
    portfolio = vbt.Portfolio.from_holding(close)
    total_return = portfolio.total_return()
    sharp_ratio = portfolio.sharpe_ratio(freq='D')  
    return total_return, sharp_ratio



