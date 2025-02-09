import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from statsmodels.api import OLS

def compute_dt(datetime_array):
    dt_seconds = np.diff(datetime_array)
    dt_years = dt_seconds / (252 * 24 * 3600)
    mean_dt = np.mean(dt_years)
    dt_years = np.insert(dt_years, 0, mean_dt)
    return dt_years

def method_moment(X, dt):
    X = np.ravel(X)
    deltaX = np.diff(X)
    X = X[:-1]
    mu = X.mean()
    exog = (mu - X) * dt
    model = OLS(endog=deltaX, exog=exog)
    res = model.fit()
    theta = res.params[0]
    resid = deltaX - theta * exog
    sigma = resid.std() / np.sqrt(dt)
    return theta, mu, sigma

def s_score(X, theta, mu, sigma):
    sigma_eq = sigma / np.sqrt(2 * theta)
    return (X - mu) / sigma_eq

def adjust_s_score(s_score, adf_p_value):
    adjustment_factor = 0.1 + (2.0 - 0.1) * (1 - np.exp(-5 * (1 - adf_p_value)))
    adjusted_s_score = s_score * adjustment_factor
    return adjusted_s_score

def compute_adf_test(x, windows):
    x = np.asarray(x)
    if np.count_nonzero(~np.isnan(x)) >= windows / 2:
        x_clean = x[~np.isnan(x)]
        result = adfuller(x_clean)
        return result[1]
    else:
        return 1

def compute_dynamic_volatility(series, model_type="EGARCH"):
    series = np.log(series).diff().dropna()
    if model_type == "EGARCH":
        model = arch_model(series, vol="EGARCH", p=1, q=1, rescale=False)
    else:
        model = arch_model(series, vol="Garch", p=1, q=1, rescale=False)
    res = model.fit(disp="off")
    conditional_vol = res.conditional_volatility
    return conditional_vol

def apply_volatility_brake(s_score, sigma_t, lambda_=0.5):
    volatility_brake = np.exp(-lambda_ * sigma_t)
    return s_score * volatility_brake

def mean_reversion(close, datetime, windows, frein, lambda_, modele):
    try:
        if frein not in ["ADF", "VOL"]:
            raise ValueError(f"Paramètre 'frein' invalide : {frein}. Les valeurs acceptées sont 'ADF' ou 'VOL'.")

        datetime = pd.to_datetime(datetime)
        dt_array = datetime.astype(np.int64) // 10**9  # Utiliser astype au lieu de view
        dt_result = compute_dt(dt_array)
        dt_series = pd.Series(dt_result, index=close.index)
        total_iterations = len(close) - windows

        with tqdm_joblib(tqdm(desc="Estimation des paramètres OU", total=total_iterations)):
            params = Parallel(n_jobs=-1)(
                delayed(method_moment)(close.iloc[i:i + windows], dt_series.iloc[i])
                for i in range(total_iterations)
            )

        theta_vals, mu_vals, sigma_vals = zip(*params)
        theta_vals = [np.nan] * windows + list(theta_vals)
        mu_vals = [np.nan] * windows + list(mu_vals)
        sigma_vals = [np.nan] * windows + list(sigma_vals)

        value_s_score = [
            s_score(close.iloc[i], theta_vals[i], mu_vals[i], sigma_vals[i])
            for i in range(windows, len(close))
        ]
        s = [np.nan] * windows + value_s_score

        if frein == "ADF":
            with tqdm_joblib(tqdm(desc="Test adf (stationarité des séries temporelles)", total=total_iterations)):
                adf_p = Parallel(n_jobs=-1)(
                    delayed(compute_adf_test)(s[i:i + windows], windows)
                    for i in range(total_iterations)
                )

            adjusted_s_scores = [
                adjust_s_score(value_s_score[i], adf_p[i])
                for i in range(len(value_s_score))
            ]

            adjusted_s_scores_adf = [np.nan] * windows + adjusted_s_scores
            s_score_value = pd.Series(s)
            adjusted_s_scores_adf = pd.Series(adjusted_s_scores_adf, index=s_score_value.index)

            return [s_score_value, adjusted_s_scores_adf, theta_vals, mu_vals,sigma_vals]

        if frein == "VOL":
            with tqdm(desc="Calcul de la volatilité conditionnelle", total=len(close)) as pbar:
                conditional_vol = compute_dynamic_volatility(close, modele)
                pbar.update(len(close))

            adjusted_s_scores_vol = [
                apply_volatility_brake(value_s_score[i], conditional_vol.iloc[i], lambda_)
                for i in range(len(value_s_score))
            ]

            adjusted_s_scores_vol = [np.nan] * windows + adjusted_s_scores_vol
            s_score_value = pd.Series(s)
            adjusted_s_scores_vol = pd.Series(adjusted_s_scores_vol, index=s_score_value.index)

            return [s_score_value, adjusted_s_scores_vol, theta_vals, mu_vals, sigma_vals]

    except ValueError as e:
        print(f"Erreur : {e}")
        return None
