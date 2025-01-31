import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import market_regime as mr


data = pd.read_csv(r'H:\Desktop\Data\FDAX_TOTAL_4H.csv')
print(data['close'])


# J'aimerais faire un test, applique l'indicateur a un actif et print sur un graphique quand l'indicateur est au dessus de 75 ou egal
windows = 150  
data['market_regime'] = mr.market_regime(data['close'], windows)

# Créer les graphiques interactifs


highlight_zones = []
in_zone = False
for i in range(len(data)):
    if data['market_regime'].iloc[i] > 74 and not in_zone:
        start = i
        in_zone = True
    elif data['market_regime'].iloc[i] <= 74 and in_zone:
        end = i
        highlight_zones.append((start, end))
        in_zone = False
if in_zone:
    highlight_zones.append((start, len(data) - 1))  # Ajouter la dernière zone si elle n'est pas fermée

# Créer une figure avec deux sous-graphiques
fig = make_subplots(
    rows=2, cols=1,  # Deux sous-graphiques, une colonne
    shared_xaxes=True,  # Partage de l'axe X entre les deux graphiques
    vertical_spacing=0.1,  # Espacement vertical entre les graphiques
    subplot_titles=('Prix de l’actif', 'Indicateur Market Regime')
)

# Sous-graphe pour les prix
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data['close'],
        mode='lines',
        name='Prix',
        line=dict(color='blue'),
    ),
    row=1, col=1  # Ajouter au premier sous-graphe
)

# Ajouter les zones surlignées au graphique des prix
for start, end in highlight_zones:
    fig.add_shape(
        type='rect',
        x0=start,
        x1=end,
        y0=data['close'].min(),
        y1=data['close'].max(),
        fillcolor='rgba(255, 0, 0, 0.2)',  # Couleur rouge semi-transparente
        line=dict(width=0),
        row=1, col=1  # Appliquer au premier sous-graphe
    )

# Sous-graphe pour l'indicateur
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data['market_regime'],
        mode='lines',
        name='Market Regime',
        line=dict(color='orange'),
    ),
    row=2, col=1  # Ajouter au deuxième sous-graphe
)

# Ajouter une ligne horizontale pointillée rouge sur le deuxième graphique
fig.add_shape(
    type='line',
    x0=data.index.min(),
    x1=data.index.max(),
    y0=73,
    y1=73,
    line=dict(
        color='red',
        width=2,
        dash='dot'
    ),
    row=2, col=1  # S'assurer qu'elle est ajoutée au deuxième sous-graphe
)

# Mettre à jour la mise en page
fig.update_layout(
    title='Prix et Indicateur Market Regime avec zones surlignées',
    xaxis_title='Index',
    yaxis_title='Prix',
    yaxis2=dict(title='Market Regime', range=[0, 100]),
    template='plotly_white',
    height=800,  # Hauteur totale de la figure
)

# Afficher le graphique
fig.show()