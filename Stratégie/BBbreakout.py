import vectorbt as vbt
import pandas as pd

###################################
#Fonction indicateur
###################################


def bb(close,period_n):
    midle = vbt.MA.run(close,window=period_n).ma
    close = pd.DataFrame(close)
    std = close.rolling(window=period_n).std()
    upper = midle + 2 * std.values 
    lower = midle - 2 * std.values
    print(upper.shape)
    print(lower.shape)
    print(midle.shape)
    return lower , midle , upper

###################################
#Fonction entrée et sortie
###################################

def entries():
    #Renvoie long_entries et short_entries
    #tu rentres en long si tu break la upper 
    #tu rentres en short si tu breaks la lower 
    pass


def exits():
    # Exemple avec long : Si le prix baisse d'un certain pourcentage k par rapport au point le plus haut atteint depuis l'ouverture
    # de la position, la position sera fermée
    # Analogue pour short 
    pass

###################################
#Indicator Factory
###################################

Ind_bb = vbt.IndicatorFactory(
    class_name= 'BB',
    input_names=['close'],
    param_names=['period_n'],
    output_names=['lower','midle','upper'],

).from_apply_func(bb)



###################################
#Strategie class
###################################


class Strategie():



    def __init__(self):
        pass


    def backtest():
        pass

    def make_values(self,close,period_n):
        Bolinger_band = Ind_bb.run(close,period_n)
        value = [Bolinger_band.lower,Bolinger_band.midle,Bolinger_band.upper]

        return value
        

    def make_entries():
        pass

    def make_exits():
        pass


strat = Strategie()


df = pd.read_csv(r'H:\Desktop\Data\USDJPY_H1.csv')
close = df['close']
BB = strat.make_values(close,20)

