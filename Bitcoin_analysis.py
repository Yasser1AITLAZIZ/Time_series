from binance.client import Client
import ta
import pandas as pd
from pandas_profiling import ProfileReport
# Initialisez le client
client = Client()

# Obtenez les informations de l'échange
exchange_info = client.get_exchange_info()

# Récupérez la liste des paires de trading
symbols = exchange_info['symbols']

# Créez une liste pour stocker les crypto-monnaies uniques
crypto_list = set()

# Parcourez chaque paire de trading et ajoutez les crypto-monnaies à la liste
for symbol_info in symbols:
    crypto_list.add(symbol_info['baseAsset'])
    crypto_list.add(symbol_info['quoteAsset'])

# Convertissez l'ensemble en liste
crypto_list = list(crypto_list)

# Affichez la liste des crypto-monnaies
print("Nombre de crypto disponible pour l'analyse :",len(crypto_list))
print(crypto_list)

def fonction_show_data(crypto= "BTCUSDT", init_date = "01 January 2017"):
    """
    Récupère les données historiques des bougies pour une crypto-monnaie spécifique depuis une date initiale.
    
    Paramètres:
    - crypto: La paire de trading pour laquelle récupérer les données (par défaut : "BTCUSDT").
    - init_date: La date à partir de laquelle commencer à récupérer les données (par défaut : "01 January 2017").
    
    Retour:
    - df: DataFrame contenant les données historiques des bougies avec les colonnes suivantes :
        * timestamp: Temps d'ouverture de la bougie en millisecondes.
        * open: Prix d'ouverture de la bougie.
        * high: Prix le plus élevé pendant la bougie.
        * low: Prix le plus bas pendant la bougie.
        * close: Prix de clôture de la bougie.
        * volume: Volume total de l'actif de base échangé pendant la bougie. L'actif de base est la première monnaie de la paire.
        * close_time: Temps de clôture de la bougie en millisecondes.
        * quote_av: Volume total de l'actif de cotation échangé pendant la bougie. L'actif de cotation est la seconde monnaie de la paire et sert de référence pour évaluer l'actif de base.
        * trades: Nombre total de trades pendant la bougie.
        * tb_base_av: Volume de l'actif de base "taker".
        * tb_quote_av: Volume de l'actif de cotation "taker".
        * ignore: Champ généralement non utilisé.
        
    Note:
    Dans une paire de trading, l'actif de cotation est la monnaie ou l'actif qui sert de référence pour évaluer ou exprimer la valeur de l'actif de base.
    """
    klinesT = Client().get_historical_klines(crypto, Client.KLINE_INTERVAL_1HOUR, init_date)
    
    df = pd.DataFrame(klinesT, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    
    # Convertir le temps en datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

    # Convertir les colonnes de type float en float
    float_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_av', 'tb_base_av', 'tb_quote_av']
    for col in float_columns:
        df[col] = df[col].astype(float)

    df.drop('ignore',axis=1,inplace=True)

    return df

data_BTCUSD = fonction_show_data()
data_BTCUSD
data_BTCUSD.info()
import plotly.graph_objects as go
def moving_average_analysis(df, window_size_1=50, window_size_2=150, window_size_3=300 ,window_size_4=500,  window_size_5=600):
    """
    Analyse de la Moyenne Mobile.
    
    La moyenne mobile est une technique de lissage des données. Elle est calculée en prenant la moyenne arithmétique
    d'un nombre défini de points de données consécutifs. Mathématiquement, une moyenne mobile simple (SMA) pour un point
    de données spécifique est donnée par :
    
    SMA = (P_t + P_(t-1) + ... + P_(t-n+1)) / n
    
    où P est le prix à l'instant t et n est le nombre de périodes.
    
    Paramètres:
    - df: DataFrame contenant les données de prix.
    - window_size: Taille de la fenêtre pour la moyenne mobile (par défaut : 50).
    """
    # Création du graphique avec Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], mode='lines', name='Prix de Clôture', line=dict(color='red')))
    for window_size in [window_size_1, window_size_2, window_size_3 ,window_size_4,  window_size_5]:
        df[f'Moving_Avg_{window_size}'] = df['close'].rolling(window=window_size).mean()
        # Ajout des traces pour le prix de clôture et la moyenne mobile
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df[f'Moving_Avg_{window_size}'], mode='lines', name=f'Moyenne Mobile {window_size}H'))
        
    fig.update_layout(
        title=f'Analyse de la Moyenne Mobile pour {window_size}H',
        xaxis_title='Date',
        yaxis_title='Prix',
        plot_bgcolor='Black',  # Couleur de fond de la zone de tracé
        paper_bgcolor='Black',  # Couleur de fond de la figure entière
        font=dict(color='gray'),  # Couleur du texte pour les titres et les étiquettes
        xaxis=dict(showgrid=True, zeroline=False, showline=True, linecolor='gray', tickfont=dict(color='White')),
        yaxis=dict(showgrid=True, zeroline=False, showline=True, linecolor='gray', tickfont=dict(color='White'))
    )
    
    # Affichage du graphique
    fig.show()


moving_average_analysis(data_BTCUSD)
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller

def test_stationarity(df):
    """
    Vérifie la stationnarité d'une série temporelle à l'aide du test de Dickey-Fuller augmenté.
    
    Paramètres:
    - df: pd.DataFrame - Le DataFrame contenant les colonnes 'timestamp' et 'close'.
    
    Retour:
    - None: Affiche le résultat du test et un graphique de la série temporelle.
    """
    
    # Assurez-vous que le DataFrame est trié par temps
    df = df.sort_values(by='timestamp')
    
    ts = df.set_index('timestamp')['close']
    
    # Calcul des statistiques mobiles
    rolmean = ts.rolling(window=12).mean()
    rolstd = ts.rolling(window=12).std()

    # Création du graphique avec Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts, mode='lines', name='Série Originale'))
    fig.add_trace(go.Scatter(x=rolmean.index, y=rolmean, mode='lines', name='Moyenne Mobile'))
    fig.add_trace(go.Scatter(x=rolstd.index, y=rolstd, mode='lines', name='Écart-type Mobile'))
    
    fig.update_layout(
    title='Moyenne Mobile & Écart-type Mobile',
    xaxis_title='Date',
    yaxis_title='Prix',
    plot_bgcolor='Black',  # Couleur de fond de la zone de tracé
    paper_bgcolor='Black',  # Couleur de fond de la figure entière
    font=dict(color='gray'),  # Couleur du texte pour les titres et les étiquettes
    xaxis=dict(showgrid=True, zeroline=False, showline=True, linecolor='gray', tickfont=dict(color='White')),
    yaxis=dict(showgrid=True, zeroline=False, showline=True, linecolor='gray', tickfont=dict(color='White')))
    fig.show()

    # Test de Dickey-Fuller :
    print('Résultats du test de Dickey-Fuller :')
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistique','p-value','#Lags utilisés','Nombre d’observations utilisées'])
    for key, value in dftest[4].items():
        dfoutput['Valeur critique (%s)' % key] = value
    print(dfoutput)



test_stationarity(data_BTCUSD)
from statsmodels.tsa.stattools import acf
import plotly.graph_objects as go

def plot_autocorrelation(df, lags=40, colonne_cible='close'):
    """
    Affiche la fonction d'autocorrélation (ACF) pour une série temporelle.
    
    L'ACF est utilisée pour identifier la saisonnalité en observant des pics significatifs 
    à des décalages spécifiques.
    
    Paramètres:
    - df: pd.DataFrame - Le DataFrame contenant les colonnes 'temps' et 'valeurs'.
    - lags: int - Le nombre de décalages à considérer pour l'ACF.
    """
    
    df = df.sort_values(by='timestamp')
    
    ts = df.set_index('timestamp')[colonne_cible]
    acf_values = acf(ts, nlags=lags, fft=True)
    
    fig = go.Figure(data=[go.Bar(x=list(range(lags+1)), y=acf_values)])
    fig.update_layout(
    title='Fonction d\'Autocorrélation (ACF)',
    xaxis_title='Décalage',
    yaxis_title='ACF',
    plot_bgcolor='Black',  # Couleur de fond de la zone de tracé
    paper_bgcolor='Black',  # Couleur de fond de la figure entière
    font=dict(color='gray'),  # Couleur du texte pour les titres et les étiquettes
    xaxis=dict(showgrid=True, zeroline=False, showline=True, linecolor='gray', tickfont=dict(color='White')),
    yaxis=dict(showgrid=True, zeroline=False, showline=True, linecolor='gray', tickfont=dict(color='White')))
    fig.show()

plot_autocorrelation(data_BTCUSD,lags=100)
data_BTCUSD_diff = data_BTCUSD.copy()
data_BTCUSD_diff['valeurs_diff'] = data_BTCUSD_diff['close'].diff()
data_BTCUSD_diff.dropna(inplace=True)  # Supprimez la première ligne car elle sera NaN

# visualisation de la série différenciée : 

fig = go.Figure()
fig.add_trace(go.Scatter(x=data_BTCUSD_diff['timestamp'], y=data_BTCUSD_diff['close'], mode='lines', name='Série Originale'))
fig.add_trace(go.Scatter(x=data_BTCUSD_diff['timestamp'] ,y=data_BTCUSD_diff['valeurs_diff'], mode='lines', name='Série différenciée'))

fig.update_layout(
title='Moyenne Mobile & Écart-type Mobile',
xaxis_title='Date',
yaxis_title='Prix',
plot_bgcolor='Black',  # Couleur de fond de la zone de tracé
paper_bgcolor='Black',  # Couleur de fond de la figure entière
font=dict(color='gray'),  # Couleur du texte pour les titres et les étiquettes
xaxis=dict(showgrid=True, zeroline=False, showline=True, linecolor='gray', tickfont=dict(color='White')),
yaxis=dict(showgrid=True, zeroline=False, showline=True, linecolor='gray', tickfont=dict(color='White')))
fig.show()

plot_autocorrelation(data_BTCUSD_diff,lags=40,colonne_cible='valeurs_diff')
data_BTCUSD_diff['valeurs_log'] = np.log(data_BTCUSD_diff['close'])
data_BTCUSD_diff['valeurs_log_diff'] = data_BTCUSD_diff['valeurs_log'].diff()
data_BTCUSD_diff.dropna(inplace=True)

# visualisation de la série différenciée : 

fig = go.Figure()
fig.add_trace(go.Scatter(x=data_BTCUSD_diff['timestamp'], y=data_BTCUSD_diff['close'], mode='lines', name='Série Originale'))
fig.add_trace(go.Scatter(x=data_BTCUSD_diff['timestamp'] ,y=data_BTCUSD_diff['valeurs_diff'], mode='lines', name='Série différenciée'))
fig.add_trace(go.Scatter(x=data_BTCUSD_diff['timestamp'] ,y=data_BTCUSD_diff['valeurs_log_diff'], mode='lines', name='Série log diff'))

fig.update_layout(
title='Moyenne Mobile & Écart-type Mobile',
xaxis_title='Date',
yaxis_title='Prix',
plot_bgcolor='Black',  # Couleur de fond de la zone de tracé
paper_bgcolor='Black',  # Couleur de fond de la figure entière
font=dict(color='gray'),  # Couleur du texte pour les titres et les étiquettes
xaxis=dict(showgrid=True, zeroline=False, showline=True, linecolor='gray', tickfont=dict(color='White')),
yaxis=dict(showgrid=True, zeroline=False, showline=True, linecolor='gray', tickfont=dict(color='White')))
fig.show()
plot_autocorrelation(data_BTCUSD_diff,lags=40,colonne_cible='valeurs_log_diff')
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go

def decompose_time_series(df, period=24, colonne_cible='close'):
    """
    Décompose une série temporelle en ses composantes de tendance, saisonnière et résiduelle.
    
    La décomposition est utile pour identifier et visualiser la saisonnalité potentielle 
    d'une série temporelle.
    
    Paramètres:
    - df: pd.DataFrame - Le DataFrame contenant les colonnes 'temps' et 'valeurs'.
    - period: int - La période supposée de la saisonnalité.
    """
    
    df = df.sort_values(by='timestamp')
    ts = df.set_index('timestamp')[colonne_cible]
    result = seasonal_decompose(ts, model='additive', period=period)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=result.trend, mode='lines', name='Tendance'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=result.seasonal, mode='lines', name='Saisonnalité'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=result.resid, mode='lines', name='Résidus'))
    fig.update_layout()
    fig.update_layout(
    title='Décomposition de la Série Temporelle', xaxis_title='Temps', yaxis_title='Valeur',
    plot_bgcolor='Black',  # Couleur de fond de la zone de tracé
    paper_bgcolor='Black',  # Couleur de fond de la figure entière
    font=dict(color='gray'),  # Couleur du texte pour les titres et les étiquettes
    xaxis=dict(showgrid=True, zeroline=False, showline=True, linecolor='gray', tickfont=dict(color='White')),
    yaxis=dict(showgrid=True, zeroline=False, showline=True, linecolor='gray', tickfont=dict(color='White')))
    fig.show()

decompose_time_series(data_BTCUSD)
decompose_time_series(data_BTCUSD_diff,colonne_cible='valeurs_diff')
decompose_time_series(data_BTCUSD_diff,colonne_cible='valeurs_log_diff')
from statsmodels.tsa.arima.model import ARIMA

def analyze_arima(ts, order=(1,1,1)):
    """
    Analyse de séries temporelles en utilisant ARIMA.
    ARIMA:
    Combine les modèles autorégressifs (AR), les modèles de moyenne mobile (MA) et la différenciation.
    Y_t = c + phi_1 Y_{t-1} + ... + phi_p Y_{t-p} + epsilon_t (pour AR)
    Y_t = c + epsilon_t + theta_1 epsilon_{t-1} + ... + theta_q epsilon_{t-q} (pour MA)
    
    Paramètres:
    - ts: La série temporelle à analyser.
    - order: Un tuple (p,d,q) représentant les ordres pour ARIMA.
    
    Retour:
    - Résultats du modèle ARIMA.
    """
    model = ARIMA(ts, order=order)
    results = model.fit()
    return results.summary()

from arch import arch_model

def analyze_garch(ts, p=1, q=1):
    """
    Analyse de séries temporelles en utilisant GARCH.
    GARCH:
    Modélise la volatilité d'une série temporelle.
    sigma^2_t = alpha_0 + sum(alpha_i epsilon^2_{t-i}) + sum(beta_j sigma^2_{t-j})
    
    Paramètres:
    - ts: La série temporelle à analyser.
    - p, q: Les ordres pour GARCH.
    
    Retour:
    - Résultats du modèle GARCH.
    """
    model = arch_model(ts, vol='Garch', p=p, q=q)
    results = model.fit()
    return results.summary()

from statsmodels.tsa.seasonal import STL

def analyze_stl(ts, period=365):
    """
    Analyse de séries temporelles en utilisant STL.
    
    STL:
    Décompose une série temporelle en saisonnalité, tendance et résidu en utilisant LOESS.
    Y_t = S_t + T_t + R_t
    
    Paramètres:
    - ts: La série temporelle à analyser.
    - period: La période pour la saisonnalité.
    
    Retour:
    - Résultats de la décomposition STL.
    """
    stl = STL(ts, seasonal=period)
    result = stl.fit()
    return result.plot()
