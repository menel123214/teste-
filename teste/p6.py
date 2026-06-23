import os
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# paths
DATA_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DATA_PATH, 'modelo_pm25_ufpa.joblib')

def load_data():
    # carregar os csvs base
    # TODO: depois talvez colocar isso num banco, por eqt lendo direto da pasta
    df_aerosol = pd.read_csv(os.path.join(DATA_PATH, 'aerosol_ufpa_corrigido.csv'))
    df_clima = pd.read_csv(os.path.join(DATA_PATH, 'Clima_Semanal_UFPA_2018_2025.csv'))
    df_no2 = pd.read_csv(os.path.join(DATA_PATH, 'NO2_UFPA_2018_2025_CORRIGIDO (1).csv'))
    
    return df_aerosol, df_clima, df_no2

def preprocess(df_aerosol, df_clima, df_no2):
    # arruma datas
    df_aerosol['data_inicio_semana'] = pd.to_datetime(df_aerosol['data_inicio_semana'], errors='coerce')
    df_clima['data_inicio'] = pd.to_datetime(df_clima['data_inicio'], errors='coerce')
    
    df_no2['Data'] = pd.to_datetime(df_no2['Data'], errors='coerce')
    df_no2['NO2'] = pd.to_numeric(df_no2['NO2'], errors='coerce')

    # no2 agrupado por semana
    df_no2 = df_no2.dropna(subset=['NO2'])
    df_no2_semanal = df_no2.set_index('Data').resample('W-MON').agg({'NO2': ['mean', 'max']})
    df_no2_semanal.columns = ['no2_mean', 'no2_max']
    df_no2_semanal = df_no2_semanal.reset_index()
    df_no2_semanal['semana_periodo'] = df_no2_semanal['Data'].dt.to_period('W-MON')

    # junta tudo
    df = pd.merge(df_aerosol, df_clima, left_on='data_inicio_semana', right_on='data_inicio', how='inner')
    df['semana_periodo'] = df['data_inicio_semana'].dt.to_period('W-MON')
    df = pd.merge(df, df_no2_semanal[['semana_periodo', 'no2_mean', 'no2_max']], on='semana_periodo', how='left')
    
    # garante a ordem pra nao bugar as series temporais
    df = df.sort_values('data_inicio_semana')

    # temp conversion
    if 'temperature_2m' in df.columns:
        df['temp_c'] = df['temperature_2m'] - 273.15

    # vars ciclicas pro mes
    df['mes'] = df['data_inicio_semana'].dt.month
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

    # speed do vento
    df['wind_speed'] = np.sqrt(df['u_component_of_wind_10m']**2 + df['v_component_of_wind_10m']**2)

    # lags do target
    for i in range(1, 6):
        df[f'lag_{i}'] = df['absorbing_aerosol_index'].shift(i)

    df['rolling_mean_3'] = df['absorbing_aerosol_index'].rolling(3).mean()
    df['rolling_std_3'] = df['absorbing_aerosol_index'].rolling(3).std()

    # dummies
    df['is_pandemia'] = np.where((df['data_inicio_semana'] >= '2020-03-15') & (df['data_inicio_semana'] <= '2021-08-31'), 1, 0)
    df['is_ferias'] = np.where(df['mes'].isin([1, 7]), 1, 0)

    return df

def build_dataset(df):
    # cols q vao pro modelo
    features = [
        'mes_sin', 'mes_cos', 'temp_c', 'total_precipitation', 'surface_pressure', 
        'wind_speed', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
        'rolling_mean_3', 'rolling_std_3', 'no2_mean', 'no2_max', 
        'is_pandemia', 'is_ferias'
    ]

    # tira na
    # print(df.isna().sum()) # debug
    df_clean = df.dropna(subset=['absorbing_aerosol_index'] + features)
    
    X = df_clean[features]
    y = df_clean['absorbing_aerosol_index']

    return X, y, df_clean, features

def temporal_split(X, y, test_size=0.2):
    # split temporal simples
    idx = int(len(X) * (1 - test_size))
    return X.iloc[:idx], X.iloc[idx:], y.iloc[:idx], y.iloc[idx:]

def train_model(X_train, y_train):
    # n_estimators mantido alto
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

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    
    print("Metricas:")
    print("R2:  ", round(r2_score(y_test, preds), 4))
    print("MAE: ", round(mean_absolute_error(y_test, preds), 4))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_test, preds)), 4))
    print("-" * 20)
    
    return preds

def diagnostics(df):
    # print(df.head())
    print("Correlacoes target:")
    corr = df.corr(numeric_only=True)['absorbing_aerosol_index'].sort_values(ascending=False).head(15)
    print(corr)
    print("-" * 20)

def plot_importance(model, features):
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=importances.index)
    plt.title('Importancia das variaveis')
    plt.tight_layout()
    # plt.savefig('importances.png')
    plt.show()

def main():
    df_aerosol, df_clima, df_no2 = load_data()
    df_full = preprocess(df_aerosol, df_clima, df_no2)
    X, y, df_clean, feats = build_dataset(df_full)
    
    diagnostics(df_clean)
    
    X_train, X_test, y_train, y_test = temporal_split(X, y)
    
    xgb_model = train_model(X_train, y_train)
    evaluate(xgb_model, X_test, y_test)
    plot_importance(xgb_model, feats)

    joblib.dump(xgb_model, MODEL_PATH)
    print("Salvo em", MODEL_PATH)

if __name__ == "__main__":
    main()
