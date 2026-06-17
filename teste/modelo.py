import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

#arquivos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#carregamento dos dados
print("Carregando as bases de dados...")

aerosol = pd.read_csv(
    os.path.join(BASE_DIR, 'aerosol_ufpa_corrigido.csv')
)

clima = pd.read_csv(
    os.path.join(BASE_DIR, 'Clima_Semanal_UFPA_2018_2025.csv')
)

no2 = pd.read_csv(
    os.path.join(BASE_DIR, 'NO2_UFPA_2018_2025_CORRIGIDO (1).csv')
)

#datas
aerosol['data_inicio_semana'] = pd.to_datetime(
    aerosol['data_inicio_semana'],
    errors='coerce'
)

clima['data_inicio'] = pd.to_datetime(
    clima['data_inicio'],
    errors='coerce'
)

no2['Data'] = pd.to_datetime(
    no2['Data'],
    errors='coerce'
)

no2['NO2'] = pd.to_numeric(
    no2['NO2'],
    errors='coerce'
)

#dados de NO2 por semana
no2 = no2.dropna(subset=['NO2'])

no2_semanal = (
    no2
    .set_index('Data')
    .resample('W-MON')
    .agg({'NO2': ['mean', 'max']})
)

no2_semanal.columns = ['no2_mean', 'no2_max']

no2_semanal = no2_semanal.reset_index()

no2_semanal['semana_periodo'] = (
    no2_semanal['Data']
    .dt
    .to_period('W-MON')
)

#U_tabelas
df = pd.merge(
    aerosol,
    clima,
    left_on='data_inicio_semana',
    right_on='data_inicio',
    how='inner'
)

df['semana_periodo'] = (
    df['data_inicio_semana']
    .dt
    .to_period('W-MON')
)

df = pd.merge(
    df,
    no2_semanal[
        ['semana_periodo', 'no2_mean', 'no2_max']
    ],
    on='semana_periodo',
    how='left'
)

df = df.sort_values('data_inicio_semana')

#variáveis auxiliares, e conv. da Tempe. para C


if 'temperature_2m' in df.columns:
    df['temp_c'] = df['temperature_2m'] - 273.15

#representação cíclica dos meses
df['mes'] = df['data_inicio_semana'].dt.month

df['mes_sin'] = np.sin(
    2 * np.pi * df['mes'] / 12
)

df['mes_cos'] = np.cos(
    2 * np.pi * df['mes'] / 12
)

#intensidade do vento
df['wind_speed'] = np.sqrt(
    df['u_component_of_wind_10m']**2 +
    df['v_component_of_wind_10m']**2
)

#histórico recente do aerossol
for i in range(1, 6):
    df[f'lag_{i}'] = (
        df['absorbing_aerosol_index']
        .shift(i)
    )

df['rolling_mean_3'] = (
    df['absorbing_aerosol_index']
    .rolling(3)
    .mean()
)

df['rolling_std_3'] = (
    df['absorbing_aerosol_index']
    .rolling(3)
    .std()
)

#indicadores do período analisado
df['is_pandemia'] = np.where(
    (
        df['data_inicio_semana'] >= '2020-03-15'
    )
    &
    (
        df['data_inicio_semana'] <= '2021-08-31'
    ),
    1,
    0
)

df['is_ferias'] = np.where(
    df['mes'].isin([1, 7]),
    1,
    0
)

#\/\/\/\/\/\/\/
variaveis = [
    'mes_sin',
    'mes_cos',
    'temp_c',
    'total_precipitation',
    'surface_pressure',
    'wind_speed',
    'lag_1',
    'lag_2',
    'lag_3',
    'lag_4',
    'lag_5',
    'rolling_mean_3',
    'rolling_std_3',
    'no2_mean',
    'no2_max',
    'is_pandemia',
    'is_ferias'
]

#removendo linhas com valores ausentes
df = df.dropna(
    subset=['absorbing_aerosol_index'] + variaveis
)

X = df[variaveis]
y = df['absorbing_aerosol_index']

# Separação temporal: 80% treino e 20% teste
corte = int(len(X) * 0.8)

X_train = X.iloc[:corte]
X_test = X.iloc[corte:]

y_train = y.iloc[:corte]
y_test = y.iloc[corte:]

print(
    f"\nBases separadas! "
    f"Treinando o modelo com {len(X_train)} registros..."
)

#treinamento XG
modelo = XGBRegressor(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

modelo.fit(X_train, y_train)

#avaliação
y_pred = modelo.predict(X_test)

print("\n--- Resultados do Teste ---")

print(
    f"R2   : {r2_score(y_test, y_pred):.4f}"
)

print(
    f"MAE  : {mean_absolute_error(y_test, y_pred):.4f}"
)

print(
    f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}"
)

# Importância das variáveis
importancias = pd.Series(
    modelo.feature_importances_,
    index=variaveis
).sort_values(ascending=True)

plt.figure(figsize=(10, 6))

importancias.plot(
    kind='barh'
)

plt.title('Importância das Variáveis')

plt.xlabel('Peso')

plt.tight_layout()

plt.show()

#save model
modelo_path = os.path.join(
    BASE_DIR,
    'modelo_pm25_ufpa.joblib'
)

joblib.dump(
    modelo,
    modelo_path
)

print(f"\nModelo salvo em: {modelo_path}")
