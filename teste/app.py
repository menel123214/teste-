import streamlit as st
import joblib
import pandas as pd
import os
import datetime

# configuracao basica da pagina web
st.set_page_config(page_title="Predição PM2.5", page_icon="🌬️", layout="centered")

st.title("🌬️ Monitoramento de PM2.5")
st.markdown("**Projeto de Iniciação Científica - UFPA** | Modelo de Predição")
st.divider()

# o decorator st.cache_resource mantem o modelo carregado na memoria do servidor
# evitando que o arquivo de IA seja lido toda vez que o usuario clica em um botao
@st.cache_resource 
def carregar_modelo():
    caminho_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_modelo = os.path.join(caminho_atual, 'modelo_pm25_ufpa.joblib')
    return joblib.load(caminho_modelo)

try:
    modelo = carregar_modelo()
except FileNotFoundError:
    st.error("erro: arquivo 'modelo_pm25_ufpa.joblib' nao encontrado. execute o treinamento primeiro.")
    st.stop()

# construcao do formulario de entrada de dados
st.header("Parâmetros de Predição")
st.write("Insira as condições meteorológicas estimadas para prever o índice de poluição.")

col1, col2 = st.columns(2)

with col1:
    data_escolhida = st.date_input("📅 Data da Análise", datetime.date.today())
    temp_c = st.number_input("🌡️ Temperatura Média (°C)", min_value=10.0, max_value=45.0, value=28.0)
    precip = st.number_input("🌧️ Precipitação Total (mm)", min_value=0.0, value=15.0)

with col2:
    vento_u = st.number_input("💨 Vento U (m/s)", value=0.0)
    vento_v = st.number_input("💨 Vento V (m/s)", value=0.0)
    pressao = st.number_input("⏱️ Pressão na Superfície (Pa)", value=100000.0)

st.write("") 

if st.button("🚀 Gerar Predição", type="primary", use_container_width=True):
    
    # organiza as entradas do usuario exatamente com os mesmos nomes de colunas
    # e na mesma ordem que o random forest aprendeu durante o fit()
    dados_usuario = pd.DataFrame(
        [[temp_c, precip, vento_u, vento_v, pressao]], 
        columns=['temp_c', 'total_precipitation', 'u_component_of_wind_10m', 'v_component_of_wind_10m', 'surface_pressure']
    )
    
    # realiza a inferencia
    resultado = modelo.predict(dados_usuario)[0]
    
    st.divider()
    st.subheader(f"Resultado para {data_escolhida.strftime('%d/%m/%Y')}")
    st.metric(label="Índice de Aerossol Estimado", value=f"{resultado:.4f}")
    
    # logica de alerta baseada em um limiar arbitrario (ajustar conforme os dados reais do projeto)
    if resultado > 1.0:
        st.warning("⚠️ Atenção: o modelo indica alta concentração de aerossóis na atmosfera.")
    else:
        st.success("✅ Condições atmosféricas dentro da normalidade esperada.")