import os
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main():
    try:
        df_aerosol = pd.read_csv('aerosol_ufpa_corrigido.csv', sep=',')
        df_clima = pd.read_csv('Clima_Semanal_UFPA_2018_2025.csv', sep=',')
        df_no2 = pd.read_csv('NO2_UFPA_2018_2025_CORRIGIDO (1).csv', sep=None, engine='python')
        df_no2.columns = df_no2.columns.str.strip() 
    except FileNotFoundError as e:
        print(f"erro ao carregar os arquivos: {e}")
        return

    df_aerosol['data_inicio_semana'] = pd.to_datetime(df_aerosol['data_inicio_semana'], errors='coerce')
    df_clima['data_inicio'] = pd.to_datetime(df_clima['data_inicio'], errors='coerce')

    df_no2['Data'] = pd.to_datetime(df_no2['Data'], errors='coerce')
    df_no2['NO2'] = pd.to_numeric(df_no2['NO2'], errors='coerce')
    
    # agrupa por semana ignorando os dias nublados (nan)
    df_no2 = df_no2.dropna(subset=['NO2'])
    df_no2_semanal = df_no2.set_index('Data').resample('W').mean().reset_index()
    df_no2_semanal.rename(columns={'NO2': 'no2_semanal'}, inplace=True)
    
    # to_period('W') padroniza a semana e evita o desalinhamento de datas no merge
    df_no2_semanal['semana_periodo'] = df_no2_semanal['Data'].dt.to_period('W')
    df_aerosol_clima = pd.merge(df_aerosol, df_clima, left_on='data_inicio_semana', right_on='data_inicio')
    df_aerosol_clima['semana_periodo'] = df_aerosol_clima['data_inicio_semana'].dt.to_period('W')
    
    df = pd.merge(df_aerosol_clima, df_no2_semanal[['semana_periodo', 'no2_semanal']], on='semana_periodo', how='left')

    df['mes'] = df['data_inicio_semana'].dt.month
    if 'temperature_2m' in df.columns:
        df['temp_c'] = df['temperature_2m'] - 273.15
        
    # cria a variavel autoregressiva (memoria da poluicao da semana anterior)
    df = df.sort_values(by='data_inicio_semana')
    df['aerossol_semana_passada'] = df['absorbing_aerosol_index'].shift(1)
    
    # proxies booleanos para estimar a variacao do fluxo de veiculos na ufpa
    df['is_pandemia'] = np.where(
        (df['data_inicio_semana'] >= '2020-03-15') & (df['data_inicio_semana'] <= '2021-08-31'), 1, 0
    )
    df['is_ferias'] = np.where(df['data_inicio_semana'].dt.month.isin([1, 7]), 1, 0)

    features = [
        'mes', 'temp_c', 'total_precipitation', 'u_component_of_wind_10m', 
        'v_component_of_wind_10m', 'surface_pressure', 'aerossol_semana_passada',
        'no2_semanal', 'is_pandemia', 'is_ferias'
    ]
    
    # o dropna aqui remove automaticamente as semanas de 2018 sem cobertura do satelite
    df_limpo = df.dropna(subset=['absorbing_aerosol_index'] + features)
    
    if df_limpo.empty:
        print("dataframe vazio apos o merge. verifique os dados.")
        return

    corr = df_limpo[features + ['absorbing_aerosol_index']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('correlacao das variaveis')
    plt.tight_layout()
    plt.show() 

    X = df_limpo[features]
    y = df_limpo['absorbing_aerosol_index']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=1000, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print(f"\nsemanas validas no treino: {len(df_limpo)}")
    print(f"r2 score: {r2_score(y_test, y_pred):.4f}")
    
    importancias = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importancias, y=importancias.index, hue=importancias.index, palette='viridis', legend=False)
    plt.title('importancia das variaveis no modelo')
    plt.tight_layout()
    plt.show()

    joblib.dump(rf, 'modelo_pm25_ufpa.joblib')

if __name__ == "__main__":
    main()