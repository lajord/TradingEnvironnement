

###################################
#Fonction indicateur
###################################


def bb(period_n):
    # Creation d'une MA (sur les n dernieres périodes)
    # upper = MA + k*std of the closing prices (sur les n dernieres périodes)
    # lower = MA - k*std of the closing prices (sur les n dernieres périodes)
    pass

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





###################################
#Strategie class
###################################


class Strategie():



    def __init__(self):
        pass


    def backtest():
        pass

    def make_values():
        pass

    def make_entries():
        pass

    def make_exits():
        pass