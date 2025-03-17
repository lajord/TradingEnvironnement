import plotly.graph_objects as go
import plotly.io as pio  
import plotly.express as px
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import numpy as np
import vectorbt as vbt
from binance.client import Client
from datetime import datetime
pio.templates.default = "plotly_dark"



INDICE_PATH = r'C:\Users\Jordi\Desktop\Environement de developement\Data\IndiceHub\{0}.csv'
INDICE_TICK_PATH = r'C:\Users\Jordi\Desktop\Environement de developement\Data\IndiceHubTicks\{0}.csv'
FOREX_PATH = r'C:\Users\Jordi\Desktop\Environement de developement\Data\ForexHub\{0}.csv'
CRYPTO_PATH = r'C:\Users\Jordi\Desktop\Environement de developement\Data\CryptoHub\{0}.csv'

# INDICE_PATH = r'H:\Desktop\Data\{0}.csv'
# INDICE_TICK_PATH = r'H:\Desktop\Data\{0}.csv'
# FOREX_PATH = r'H:\Desktop\Data\{0}.csv'

#-------------------------------------------------DATA GESTION-------------------------------------------------#


def get_data_indice(tickers, frequences, start_period=None, end_period=None):
    data_dict = {freq: {} for freq in frequences}  
    max_length = {freq: 0 for freq in frequences} 

    for ticker in tickers:
        file_path = INDICE_PATH.format(ticker)
        
        try:
            df = pd.read_csv(file_path)
            df = uniform_convention_data(df)  # Standardiser les conventions
            
            if start_period:
                start_date = pd.to_datetime(start_period)
                df = df.loc[df.index >= start_date]
            if end_period:
                end_date = pd.to_datetime(end_period)
                df = df.loc[df.index <= end_date]

            # Appliquer le resampling pour chaque fréquence demandée
            for freq in frequences:
                df_resampled = resample_and_return(df, freq)
                data_dict[freq][ticker] = df_resampled  # Stocker le dataframe
                max_length[freq] = max(max_length[freq], len(df_resampled))  # Mettre à jour max_length
                
        except Exception as e:
            print(f"Erreur lors du chargement des données pour {ticker}: {e}")

    # Uniformisation de la taille (version corrigée)
    for freq, tickers_data in data_dict.items():
        for ticker, df in tickers_data.items():
            if len(df) < max_length[freq]:
                fill_data = {col: [np.nan] * (max_length[freq] - len(df)) for col in df.columns}
                df = pd.concat([df, pd.DataFrame(fill_data)], ignore_index=False)
                df.index.name = 'Datetime'
                data_dict[freq][ticker] = df  # Mettre à jour avec la version uniformisée

    return data_dict





def get_data_forex(tickers, frequences, start_period=None, end_period=None):
    data_dict = {freq: {} for freq in frequences}  
    max_length = {freq: 0 for freq in frequences}  

    for ticker in tickers:
        file_path = FOREX_PATH.format(ticker)
        try:
            df = pd.read_csv(file_path)
            df = uniform_convention_data(df)  # Standardiser les conventions
            
            if start_period:
                start_date = pd.to_datetime(start_period)
                df = df.loc[df.index >= start_date]
            if end_period:
                end_date = pd.to_datetime(end_period)
                df = df.loc[df.index <= end_date]

            # Appliquer le resampling pour chaque fréquence demandée
            for freq in frequences:
                df_resampled = resample_and_return(df, freq)
                data_dict[freq][ticker] = df_resampled  # Stocker le dataframe
                max_length[freq] = max(max_length[freq], len(df_resampled))  # Mettre à jour max_length
                
        except Exception as e:
            print(f"Erreur lors du chargement des données pour {ticker}: {e}")

    # Uniformiser les tailles pour chaque timeframe
    for freq, tickers_data in data_dict.items():
        for ticker, df in tickers_data.items():
            if len(df) < max_length[freq]:
                fill_data = {col: [np.nan] * (max_length[freq] - len(df)) for col in df.columns}
                df = pd.concat([df, pd.DataFrame(fill_data)], ignore_index=False)
                df.index.name = 'Datetime'
                data_dict[freq][ticker] = df  # Mettre à jour avec la version uniformisée

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


def get_data_crypto(tickers, frequences, start_period=None, end_period=None):
    data_dict = {freq: {} for freq in frequences}  
    max_length = {freq: 0 for freq in frequences}  

    for ticker in tickers:
        file_path = CRYPTO_PATH.format(ticker)
        try:
            df = pd.read_csv(file_path)
            df = uniform_convention_data(df)  # Standardiser les conventions
            
            if start_period:
                start_date = pd.to_datetime(start_period)
                df = df.loc[df.index >= start_date]
            if end_period:
                end_date = pd.to_datetime(end_period)
                df = df.loc[df.index <= end_date]

            # Appliquer le resampling pour chaque fréquence demandée
            for freq in frequences:
                df_resampled = resample_and_return(df, freq)
                data_dict[freq][ticker] = df_resampled  # Stocker le dataframe
                max_length[freq] = max(max_length[freq], len(df_resampled))  # Mettre à jour max_length
                
        except Exception as e:
            print(f"Erreur lors du chargement des données pour {ticker}: {e}")

    # Uniformiser les tailles pour chaque timeframe
    for freq, tickers_data in data_dict.items():
        for ticker, df in tickers_data.items():
            if len(df) < max_length[freq]:
                fill_data = {col: [np.nan] * (max_length[freq] - len(df)) for col in df.columns}
                df = pd.concat([df, pd.DataFrame(fill_data)], ignore_index=False)
                df.index.name = 'Datetime'
                data_dict[freq][ticker] = df  # Mettre à jour avec la version uniformisée

    return data_dict



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











def resample_shape_frequence(referenciel, modifier):
    original_index_type = type(modifier.index)  
    if not isinstance(referenciel.index, pd.DatetimeIndex):
        referenciel.index = pd.to_datetime(referenciel.index)
    if not isinstance(modifier.index, pd.DatetimeIndex):
        modifier.index = pd.to_datetime(modifier.index)

    if not isinstance(referenciel.index, pd.DatetimeIndex):
        raise ValueError("L'index de `referenciel` n'est pas un DatetimeIndex après conversion.")
    if not isinstance(modifier.index, pd.DatetimeIndex):
        raise ValueError("L'index de `modifier` n'est pas un DatetimeIndex après conversion.")


    if len(modifier) > len(referenciel):
        raise ValueError("Le dataframe `modifier` contient plus de valeurs que `referenciel`, ce qui est incohérent.")


    taille_initiale = len(referenciel)
    referenciel_clean = referenciel.dropna(how='all').sort_index()
    modifier_clean = modifier.dropna(how='all').sort_index()
    modifier_aligned = modifier_clean.reindex(referenciel_clean.index, method='ffill')
    try:
        modifier_aligned.index = original_index_type(modifier_aligned.index)
    except Exception as e:
        print(f"Impossible de restaurer l'index original ({original_index_type}): {e}")
    if len(modifier_aligned) < taille_initiale:
        nan_pad = pd.DataFrame(index=referenciel.index[len(modifier_aligned):], columns=modifier_aligned.columns)
        modifier_aligned = pd.concat([modifier_aligned, nan_pad])

    return modifier_aligned








#------------------------------------------------------------------------------------------------------------------#







#---------------------------------------------------PRINT ELEMENT--------------------------------------------------#

"""
Precondition : Un dataframe OHLC,  un dataframe avec au moins un side c'est a dire entries et short
Postcondition : print un graphique movible
"""
def print_trades(entries_long, entries_short, exits_long, exits_short, data):
    numeric_index = np.arange(len(data['close']))  

    fig = go.Figure(data=[go.Scatter(
        x=numeric_index,
        y=data['close'],
        mode='lines',
        name='Prix de clôture',
        line=dict(color='#00ff88', width=1)
    )])


    def validate_and_convert(series, reference):
        """ Vérifie que series est bien un array de booléens et a la bonne taille """
        if series is None:
            return np.array([], dtype=int)  
        series = np.asarray(series)  
        if len(series) != len(reference):  
            raise ValueError(f"Longueur invalide : attendu {len(reference)}, reçu {len(series)}")
        return np.where(series)[0]  

    try:
        entries_long_idx = validate_and_convert(entries_long, data['close'])
        entries_short_idx = validate_and_convert(entries_short, data['close'])
        exits_long_idx = validate_and_convert(exits_long, data['close'])
        exits_short_idx = validate_and_convert(exits_short, data['close'])
    except ValueError as e:
        print(f"Erreur : {e}")
        return  


    if len(entries_long_idx) > 0:
        fig.add_trace(go.Scatter(
            x=numeric_index[entries_long_idx],
            y=data['close'].iloc[entries_long_idx],
            mode='markers',
            marker=dict(symbol='x', size=10, color='blue'),
            name='Entrée Long'
        ))


    if len(entries_short_idx) > 0:
        fig.add_trace(go.Scatter(
            x=numeric_index[entries_short_idx],
            y=data['close'].iloc[entries_short_idx],
            mode='markers',
            marker=dict(symbol='x', size=10, color='red'),
            name='Entrée Short'
        ))


    if len(exits_long_idx) > 0:
        fig.add_trace(go.Scatter(
            x=numeric_index[exits_long_idx],
            y=data['close'].iloc[exits_long_idx],
            mode='markers',
            marker=dict(symbol='triangle-up', size=10, color='blue'),
            name='Sortie Long'
        ))


    if len(exits_short_idx) > 0:
        fig.add_trace(go.Scatter(
            x=numeric_index[exits_short_idx],
            y=data['close'].iloc[exits_short_idx],
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
    fig.update_layout(template="plotly_dark", showlegend=False)  
    fig.show()





#-----------------------------------------------------------------------------------------------------------------#










#------------------------------------------------------METRICS----------------------------------------------------#


def get_details_variable(liste):
    for var in liste:
        print(f"Variable : {getattr(var, 'name', 'N/A')}")  # Utilise 'N/A' si .name n'existe pas
        print(f"Type : {type(var)}")
        
        # Vérifie si l'objet a un attribut `.shape`
        if hasattr(var, "shape"):
            print(f"Shape : {var.shape}")
        elif isinstance(var, list):  # Si c'est une liste, affiche la longueur
            print(f"Shape (approximatif) : ({len(var)},)")
        else:
            print("Shape : N/A")
        
        print("\n")


def get_metrics(portfolio,params):
    if params == 1:
        return portfolio.total_return().mean()
    if params == 2:
        return portfolio.sharpe_ratio().mean()
    if params == 3:
        return portfolio.calmar_ratio().mean()
    if params == 4:
        return  portfolio.win_rate().mean()

def get_3d_surface_metrics(params,portfolio, combinaison, metrics,axex,axey, elev=30, azim=120):
    perf_combi = []
    if params==2:
        for index in range(len(combinaison)):  
            metrics_avrg = []  

            value_metrics = get_metrics(portfolio[index], metrics)
            metrics_avrg.append(round(value_metrics,2))
                
            avg_metrics = sum(metrics_avrg)  
            param1, param2 = combinaison[index]  
            perf_combi.append((param1, param2, avg_metrics))  

        df_perf = pd.DataFrame(perf_combi, columns=["Paramètre 1", "Paramètre 2", "Performance"])
        df_perf = df_perf.groupby(["Paramètre 1", "Paramètre 2"]).mean().reset_index()
        df_pivot = df_perf.pivot(index="Paramètre 2", columns="Paramètre 1", values="Performance")

        X = df_pivot.columns.values  
        Y = df_pivot.index.values    
        X, Y = np.meshgrid(X, Y)  
        Z = df_pivot.values       

        # Création de la figure et de l'axe 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Définition de l'angle de vue
        ax.view_init(elev=elev, azim=azim)

        # Tracé de la surface avec `plot_surface`
        surf = ax.plot_surface(X, Y, Z, cmap="RdYlGn", edgecolor='k', linewidth=0.5)
        ax.set_xlabel(axex)
        ax.set_ylabel(axey)
        ax.set_zlabel("Performance")
        ax.set_title("Graphique 3D des performances")

        # Ajout d'une barre de couleur
        fig.colorbar(surf, shrink=0.5, aspect=10)

        plt.show()
    else:
        perf_combi = []

        for index in range(len(combinaison)):  
            param1, param2, param3 = combinaison[index]  
            value_metrics = get_metrics(portfolio[index], metrics)
            perf_combi.append((param1, param2, param3, round(value_metrics, 2)))  

        df_perf = pd.DataFrame(perf_combi, columns=["Paramètre 1", "Paramètre 2", "Paramètre 3", "Performance"])

        X = df_perf["Paramètre 1"].values  
        Y = df_perf["Paramètre 2"].values  
        Z = df_perf["Paramètre 3"].values  
        P = df_perf["Performance"].values  

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.view_init(elev=elev, azim=azim)

        # Tracé de la surface avec trisurf
        surf = ax.plot_trisurf(X, Y, P, cmap="RdYlGn", edgecolor='k', linewidth=0.5)
        
        ax.set_xlabel("Paramètre 1")
        ax.set_ylabel("Paramètre 2")
        ax.set_zlabel("Performance")
        ax.set_title("Graphique 3D des performances")

        fig.colorbar(surf, shrink=0.5, aspect=10)

        plt.show()





def get_barplot(portfolio, combinaison, metrics,params_name, figsize_x=10, figsize_y=6):
    perf_combi = []  
    for index in range(len(combinaison)):  
        value_metrics = get_metrics(portfolio[index], metrics)
        perf_combi.append((combinaison[index], round(value_metrics, 2)))  
    df_perf = pd.DataFrame(perf_combi, columns=["Paramètre", "Performance"])
    plt.figure(figsize=(figsize_x, figsize_y))  # Taille ajustable
    sns.barplot(data=df_perf, x="Paramètre", y="Performance", palette="RdYlGn")
    plt.xlabel(params_name)
    plt.ylabel("Performance")
    plt.title("Performance en fonction du paramètre")
    plt.xticks(rotation=45, fontsize=max(8, figsize_x))  
    plt.yticks(fontsize=max(8, figsize_y))  
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()





#Print une heatmap des performance 
def get_heatmap(portfolio, combinaison, metrics, params1_name, params2_name, figsize_x=10, figsize_y=8):
    perf_combi = []
    for index in range(len(combinaison)):  
        value_metrics = get_metrics(portfolio[index], metrics)
        param1, param2 = combinaison[index]  
        perf_combi.append((param1, param2, round(value_metrics, 2)))  
    df_perf = pd.DataFrame(perf_combi, columns=[params1_name, params2_name, "Performance"])
    df_perf = df_perf.groupby([params1_name, params2_name]).mean().reset_index()
    df_pivot = df_perf.pivot(index=params2_name, columns=params1_name, values="Performance")
    plt.figure(figsize=(figsize_x, figsize_y))  # Taille ajustable
    sns.heatmap(df_pivot, cmap="RdYlGn", annot=True, fmt=".3f", annot_kws={"size": 4})
    plt.xlabel(params1_name, fontsize=max(10, figsize_x))
    plt.ylabel(params2_name, fontsize=max(10, figsize_y))
    plt.title("Heatmap des performances", fontsize=max(12, (figsize_x + figsize_y) // 2))
    plt.xticks(fontsize=max(8, figsize_x - 2))
    plt.yticks(fontsize=max(8, figsize_y - 2))
    plt.show()





def get_best_param(portfolio, combinaison, metrics, nbr_param):
    best_combi = []
    max = -10000

    if nbr_param ==1 :
        for index in range(len(combinaison)):   
            value_metrics = get_metrics(portfolio[index], metrics)
            param1 = combinaison[index]
            if value_metrics > max:
                max = value_metrics
                best_combi = param1


    if nbr_param == 2:
        for index in range(len(combinaison)):  
            value_metrics = get_metrics(portfolio[index], metrics)
            param1, param2 = combinaison[index]  
            if value_metrics > max:
                max = value_metrics
                best_combi = [param1,param2]
    
    return best_combi


def generate_metrics_report(portfolio, close,ticker):
    benchmark_return, benchmark_sharp = get_metrics_buy_and_hold(close[ticker])
    port = portfolio[ticker]

    print(f"Total Return : {round(port.total_return() * 100, 2)}%")
    print(f'Benchmark return : {round(benchmark_return*100,2)}%')
    print(f"CAGR : {round(port.annualized_return()*100,2)}%")
    print(f"Volatility : {round(port.annualized_volatility()*100,2)}%")
    print(f"Sharpe Ratio : {round(port.sharpe_ratio(),2)}")
    print(f'Benchmark sharpe ratio : {round(benchmark_sharp,2)}')
    print(f"Max Drawdown: {round(port.max_drawdown()*100,2)}%")
    print(f"Calmar Ratio : {round(port.calmar_ratio(),2)}")
    print(f"Beta : {round(port.beta(),2)}")
    print('\n')
    print(f'Total trades : {round(port.trades.count(),0)}')
    print(f"Total long trades {port.trades.long.count()}")
    print(f"Total short trades {port.trades.short.count()}")
    print(f"Win rate : {round(port.trades.closed.win_rate(),2)}%")
    print(f"Wining streak : {port.trades.winning_streak.max()}")
    print(f"Loosing streak : {port.trades.losing_streak.max()}")
    print(f"Average winning trade : {round(port.trades.winning.returns.mean(),2)*100}%")
    print(f"Average losing trade : {round(port.trades.losing.returns.mean(),2)*100}%")
    print(f"Profit factor : {round(port.trades.profit_factor(),2)}")
    print(f"Expectancy : {round(port.trades.expectancy(),2)}%")


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






def rapport_backtest(portfolio,close,datetime,tickers):
    if len(tickers) >1:
        get_pnl(portfolio)
        print('\n---------------------------------------------------------------GLOBAL EQUITY CURVE---------------------------------------------------------------\n')
        get_equity_global(portfolio,tickers)
        print('\n------------------------------------------------------------------GLOBAL METRICS-----------------------------------------------------------------\n')
    for ticker in tickers:
        print(f'\n------------------------------------------------------------------DETAILS : {ticker}-----------------------------------------------------------------\n')
        generate_metrics_report(portfolio,close,ticker)
        get_pnl(portfolio[ticker])
        # average_duration_and_plot(portfolio[ticker],datetime[ticker])
        # portfolio.plot_drawdowns(column=ticker).show()
        # portfolio.plot_underwater(column=ticker).show()
        print('\n--------------------------------------------------------------------------------------------------------------------------------------------------\n')


def average_duration_and_plot(port, datatime):
    exits_idx = port.trades.records_readable['Exit Timestamp']
    entries_idx = port.trades.records_readable['Entry Timestamp']
    exits_dt = datatime.iloc[exits_idx].astype('datetime64[ns]')
    entries_dt = datatime.iloc[entries_idx].astype('datetime64[ns]')
    trade_durations = (exits_dt.values - entries_dt.values) / pd.Timedelta(days=1)
    trade_durations = trade_durations[trade_durations >= 0]  # Enlever les durées négatives
    trade_returns = port.trades.records_readable['Return']
    # Vérification de la taille des séries et ajustement
    min_length = min(len(trade_durations), len(trade_returns))
    trade_durations = trade_durations[:min_length]
    trade_returns = trade_returns[:min_length]
    df = pd.DataFrame({
        "duration_days": trade_durations,
        "return": trade_returns
    })
    df["date"] = entries_dt.dt.date
    avg_duration_per_day = df['duration_days'].mean()
    # Création du graphique avec Plotly Graph Objects
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["duration_days"],
        y=df["return"],
        mode='markers',
        marker=dict(size=8, color="blue", opacity=0.7),
        name="Trade Returns"
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red", name="Neutral Return Line")  # Ligne horizontale à Y=0
    fig.update_layout(
        title="Durée des Trades vs Rendement",
        xaxis_title="Durée du trade (jours)",
        yaxis_title="Rendement du trade",
        template="plotly_white",
        showlegend=True,
        width=700,  # Ajuster la largeur du graphique
        height=500,  # Ajuster la hauteur du graphique
    )
    fig.show()
    print(f'\n -> Average duration trades (jour) : {round(avg_duration_per_day,2)}\n')
    return avg_duration_per_day




def get_equity_global(portfolio, tickers, initial_capital=100000):
    all_trades_returns = [] 
    smoothing_window=5
    for ticker in tickers:
        trade_returns = portfolio[ticker].trades.records_readable['Return']  
        all_trades_returns.append(trade_returns)
    if not all_trades_returns:
        print("Aucun trade trouvé !")
        return None
    combined_returns = pd.concat(all_trades_returns).sort_index()
    global_equity_curve = (1 + combined_returns).cumprod() * initial_capital
    smoothed_equity_curve = global_equity_curve.rolling(window=smoothing_window, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=smoothed_equity_curve.index, 
        y=smoothed_equity_curve, 
        mode='lines',
        name='Smoothed Equity Curve',
        line=dict(color='red', width=2)
    ))
    fig.update_layout(
        title="Performance du Portefeuille (Lissé)",
        xaxis_title="Trade Number",
        yaxis_title="Portfolio Value ($)",
        template="plotly_dark",
        hovermode="x",
        width=900,  
        height=500  
    )
    fig.show()
    return global_equity_curve




def generate_wf_cycles(df, ios_period, oos_period):
    cycles = []
    start_date = df.index.min()
    end_date = df.index.max()
    
    while True:
        ios_start = start_date
        ios_end = ios_start + pd.DateOffset(years=int(ios_period[:-1])) if 'Y' in ios_period else ios_start + pd.DateOffset(months=int(ios_period[:-1]))
        
        oos_start = ios_end 
        oos_end = oos_start + pd.DateOffset(years=int(oos_period[:-1])) if 'Y' in oos_period else oos_start + pd.DateOffset(months=int(oos_period[:-1]))
        
        if oos_end > end_date:
            oos_end = end_date
            ios_start_idx = df.index.get_indexer([ios_start], method='bfill')[0]
            ios_end_idx = df.index.get_indexer([ios_end], method='bfill')[0]
            oos_start_idx = df.index.get_indexer([oos_start], method='bfill')[0]
            oos_end_idx = df.index.get_indexer([oos_end], method='bfill')[0] + 1
            cycles.append((ios_start_idx, ios_end_idx, oos_start_idx, oos_end_idx))
            break
        else:

            ios_start_idx = df.index.get_indexer([ios_start], method='bfill')[0]
            ios_end_idx = df.index.get_indexer([ios_end], method='bfill')[0]
            oos_start_idx = df.index.get_indexer([oos_start], method='bfill')[0]
            oos_end_idx = df.index.get_indexer([oos_end], method='bfill')[0]
            cycles.append((ios_start_idx, ios_end_idx, oos_start_idx, oos_end_idx))
        start_date = start_date + pd.DateOffset(years=int(oos_period[:-1])) if 'Y' in oos_period else start_date + pd.DateOffset(months=int(oos_period[:-1]))
    
    return cycles

#---------------------------------------------------------------------------------------------------------------------------------------#










#------------------------------------------------------ REALITY CHECK ----------------------------------------------------#


#Cette fct prend en parametre un portfolio et extrais les trades, puis ensuite print un graphique avec un nbr de lancer
def bootstrapping(portfolio,nbr_lancer):
    trade_returns = portfolio.trades.returns
    print(trade_returns)

