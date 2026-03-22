import ee
import joblib
import pandas as pd

# inicializa a conexao com os servidores do google earth engine
ee.Initialize()

def buscar_clima_ufpa(data):
    # define o ponto central do campus
    ponto_ufpa = ee.Geometry.Point([-48.4552, -1.4735])
    
    # busca os dados climaticos do dia especifico no satelite
    colecao = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
               .filterBounds(ponto_ufpa)
               .filterDate(data, ee.Date(data).advance(1, 'day')))
    
    # calcula a media do dia e extrai os valores
    dados = colecao.mean().reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=ponto_ufpa,
        scale=11132
    ).getInfo()
    
    return dados

def prever_pm25(dados_clima):
    # carrega o mRF e realiza a predicao
    modelo = joblib.load('modelo_pm25_ufpa.joblib')
    df_input = pd.DataFrame([dados_clima])
    return modelo.predict(df_input)[0]

if __name__ == "__main__":
    data_alvo = '2026-03-20'
    
    try:
        clima_hoje = buscar_clima_ufpa(data_alvo)
        pm25_estimado = prever_pm25(clima_hoje)
        
        print(f"estimativa de pm2.5 na ufpa em {data_alvo}: {pm25_estimado:.2f}")
        
    except Exception as e:
        print(f"erro ao executar a predicao: {e}")