import os
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# =========================
# CONFIG
# =========================
DATA_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DATA_PATH, 'modelo_pm25_ufpa.joblib')

# =========================
# LOAD DATA
# =========================
def load_data():
    try:
        df_aerosol = pd.read_csv(os.path.join(DATA_PATH, 'aerosol_ufpa_corrigido.csv'))
        df_clima = pd.read_csv(os.path.join(DATA_PATH, 'Clima_Semanal_UFPA_2018_2025.csv'))
        df_no2 = pd.read_csv(os.path.join(DATA_PATH, 'NO2_UFPA_2018_2025_CORRIGIDO (1).csv'))
    except FileNotFoundError as e:
        raise RuntimeError(f"Erro ao carregar arquivos: {e}")

    return df_aerosol, df_clima, df_no2


# =========================
# PREPROCESSING
# =========================
def preprocess(df_aerosol, df_clima, df_no2):
    # Datas
    df_aerosol['data_inicio_semana'] = pd.to_datetime(df_aerosol['data_inicio_semana'], errors='coerce')
    df_clima['data_inicio'] = pd.to_datetime(df_clima['data_inicio'], errors='coerce')

    df_no2['Data'] = pd.to_datetime(df_no2['Data'], errors='coerce')
    df_no2['NO2'] = pd.to_numeric(df_no2['NO2'], errors='coerce')

    # NO2 semanal
    df_no2 = df_no2.dropna(subset=['NO2'])
    df_no2_semanal = (
        df_no2
        .set_index('Data')
        .resample('W-MON')
        .agg({'NO2': ['mean', 'max']})
    )

    df_no2_semanal.columns = ['no2_mean', 'no2_max']
    df_no2_semanal = df_no2_semanal.reset_index()
    df_no2_semanal['semana_periodo'] = df_no2_semanal['Data'].dt.to_period('W-MON')

    # Merge principal
    df = pd.merge(
        df_aerosol,
        df_clima,
        left_on='data_inicio_semana',
        right_on='data_inicio',
        how='inner'
    )

    df['semana_periodo'] = df['data_inicio_semana'].dt.to_period('W-MON')

    df = pd.merge(
        df,
        df_no2_semanal[['semana_periodo', 'no2_mean', 'no2_max']],
        on='semana_periodo',
        how='left'
    )

    df = df.sort_values('data_inicio_semana')

    # =========================
    # FEATURE ENGINEERING
    # =========================

    # Temperatura
    if 'temperature_2m' in df.columns:
        df['temp_c'] = df['temperature_2m'] - 273.15

    # Mês cíclico
    df['mes'] = df['data_inicio_semana'].dt.month
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

    # Vento (magnitude)
    df['wind_speed'] = np.sqrt(
        df['u_component_of_wind_10m']**2 +
        df['v_component_of_wind_10m']**2
    )

    # LAGS (memória temporal)
    for lag in range(1, 6):
        df[f'lag_{lag}'] = df['absorbing_aerosol_index'].shift(lag)

    # Média móvel
    df['rolling_mean_3'] = df['absorbing_aerosol_index'].rolling(3).mean()
    df['rolling_std_3'] = df['absorbing_aerosol_index'].rolling(3).std()

    # Proxies
    df['is_pandemia'] = np.where(
        (df['data_inicio_semana'] >= '2020-03-15') &
        (df['data_inicio_semana'] <= '2021-08-31'), 1, 0
    )

    df['is_ferias'] = np.where(df['mes'].isin([1, 7]), 1, 0)

    return df


# =========================
# DATASET
# =========================
def build_dataset(df):
    features = [
        'mes_sin', 'mes_cos',
        'temp_c',
        'total_precipitation',
        'surface_pressure',
        'wind_speed',
        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
        'rolling_mean_3', 'rolling_std_3',
        'no2_mean', 'no2_max',
        'is_pandemia', 'is_ferias'
    ]

    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    df = df.dropna(subset=['absorbing_aerosol_index'] + features)

    if df.empty:
        raise ValueError("Dataframe vazio após limpeza")

    X = df[features]
    y = df['absorbing_aerosol_index']

    return X, y, df, features


# =========================
# TEMPORAL SPLIT
# =========================
def temporal_split(X, y, test_size=0.2):
    split_idx = int(len(X) * (1 - test_size))

    return (
        X.iloc[:split_idx],
        X.iloc[split_idx:],
        y.iloc[:split_idx],
        y.iloc[split_idx:]
    )


# =========================
# MODEL
# =========================
def train_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=100000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=1042
    )

    model.fit(X_train, y_train)
    return model


# =========================
# EVALUATION
# =========================
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\n===== MÉTRICAS =====")
    print(f"R2   : {r2_score(y_test, y_pred):.4f}")
    print(f"MAE  : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

    return y_pred


# =========================
# DIAGNOSTICS
# =========================
def diagnostics(df):
    print("\n===== CORRELAÇÃO COM TARGET =====")
    print(
        df.corr(numeric_only=True)['absorbing_aerosol_index']
        .sort_values(ascending=False)
        .head(15)
    )


# =========================
# PLOTS
# =========================
def plot_importance(model, features):
    importancias = pd.Series(model.feature_importances_, index=features)\
        .sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importancias, y=importancias.index)
    plt.title('Importância das variáveis')
    plt.tight_layout()
    plt.show()


# =========================
# MAIN
# =========================
def main():
    df_aerosol, df_clima, df_no2 = load_data()

    df = preprocess(df_aerosol, df_clima, df_no2)

    X, y, df, features = build_dataset(df)

    diagnostics(df)

    X_train, X_test, y_train, y_test = temporal_split(X, y)

    model = train_model(X_train, y_train)

    evaluate(model, X_test, y_test)

    plot_importance(model, features)

    joblib.dump(model, MODEL_PATH)
    print(f"\nModelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()