import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Dashboard AIoT - Monitoramento Sísmico USGS", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌋 Dashboard AIoT: Monitoramento e Predição de Terremotos (Padrão USGS)")
st.markdown("""
Esta aplicação simula o nó central (Gateway/Cloud) de uma rede **AIoT (Inteligência Artificial das Coisas)**. 
Ela recebe fluxos de dados via sismógrafos baseados nas especificações oficiais do Serviço Geológico dos Estados Unidos.
""")
st.markdown("---")

# CARREGA OS MODELOS
@st.cache_resource
def load_data():
    with open('modelo/modelo_regressao_linear.pkl', 'rb') as f: lr = pickle.load(f)
    with open('modelo/modelo_random_forest.pkl', 'rb') as f: rf = pickle.load(f)
    with open('modelo/metricas.pkl', 'rb') as f: met = pickle.load(f)
    return lr, rf, met

try:
    lr_model, rf_model, metrics = load_data()
    df_metrics = pd.DataFrame(metrics)
except FileNotFoundError:
    st.error("❌ ERRO: Arquivos de modelos USGS não encontrados na pasta 'modelo/'!")
    st.info("💡 Execute o script de treinamento `train.py` para gerar as matrizes e arquivos necessários.")
    st.stop()

# SIDEBAR: ENTRADA DE DADOS SIMULADOS
st.sidebar.header("📡 Configuração da Estação IoT")

locais_predefinidos = {
    "Personalizado (Ajuste livre)": {"lat": 0.0, "lon": 0.0, "depth": 10.0, "rms": 0.4, "gap": 90.0, "sig": 150},
    "Tóquio, Japão 🇯🇵": {"lat": 35.6762, "lon": 139.6503, "depth": 40.0, "rms": 0.18, "gap": 22.0, "sig": 720},
    "Santiago, Chile 🇨🇱": {"lat": -33.4489, "lon": -70.6693, "depth": 25.0, "rms": 0.25, "gap": 35.0, "sig": 640},
    "Califórnia, EUA 🇺🇸": {"lat": 36.1749, "lon": -120.7247, "depth": 8.0, "rms": 0.12, "gap": 15.0, "sig": 380},
    "Sumatra, Indonésia 🇮🇩": {"lat": -0.5897, "lon": 101.3431, "depth": 35.0, "rms": 0.38, "gap": 75.0, "sig": 810},
    "Istambul, Turquia 🇹🇷": {"lat": 41.0082, "lon": 28.9784, "depth": 12.0, "rms": 0.21, "gap": 45.0, "sig": 490}
}

local_selecionado = st.sidebar.selectbox("Selecione a Região do Sensor:", list(locais_predefinidos.keys()))
dados_locais = locais_predefinidos[local_selecionado]

st.sidebar.markdown("---")
st.sidebar.subheader("Leituras de Telemetria do Hardware")

input_lat = st.sidebar.slider("Latitude", -90.0, 90.0, dados_locais["lat"])
input_lon = st.sidebar.slider("Longitude", -180.0, 180.0, dados_locais["lon"])
input_depth = st.sidebar.number_input("Profundidade (km)", min_value=0.0, max_value=700.0, value=dados_locais["depth"])
input_rms = st.sidebar.slider("Erro Residual (RMS do Sensor)", 0.0, 5.0, dados_locais["rms"])
input_gap = st.sidebar.slider("Ponto Cego (GAP Azimutal)", 0.0, 360.0, dados_locais["gap"])
input_sig = st.sidebar.slider("Significância Sísmica (sig)", 0, 2000, dados_locais["sig"])

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("🛠️ Simulador de Evento Sísmico (Ação da IA)")
    st.markdown("Submeta leituras de sensores captadas na borda (*Edge*) para processamento inteligente centralizado.")
    
    with st.form("form_usgs_evento"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("**Geolocalização:**")
            lat_ev = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=input_lat)
            lon_ev = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=input_lon)
            depth_ev = st.number_input("Profundidade (Foco km)", min_value=0.0, max_value=700.0, value=input_depth)
        
        with col_f2:
            st.markdown("**Métricas do Hardware IoT:**")
            rms_ev = st.number_input("Erro de Sinal (RMS)", min_value=0.0, max_value=5.0, value=input_rms)
            gap_ev = st.number_input("Abertura de Estações (GAP)", min_value=0.0, max_value=360.0, value=input_gap)
            sig_ev = st.number_input("Significância Histórica (sig)", min_value=0, max_value=2000, value=input_sig)
            
        submetido = st.form_submit_button("🌋 Analisar Risco do Evento")

    if submetido:
        st.markdown("### 🧠 Diagnóstico da Inteligência Artificial")
        
        dados_novos = pd.DataFrame([[lat_ev, lon_ev, depth_ev, rms_ev, gap_ev, sig_ev]], 
                                   columns=['latitude', 'longitude', 'depth', 'rms', 'gap', 'sig'])
        
        pred_lr = lr_model.predict(dados_novos)[0]
        pred_rf = rf_model.predict(dados_novos)[0]
        
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Previsão Regressão Linear", f"{pred_lr:.2f} Mw")
        col_res2.metric("Previsão Random Forest", f"{pred_rf:.2f} Mw")
        
        mag_final = pred_rf
        
        if mag_final >= 5.0:
            st.error(f"""
            ### 🚨 STATUS: ALERTA CRÍTICO ATIVADO
            *   **Magnitude Estimada:** {mag_final:.2f} Mw (Via Random Forest)
            *   **Risco Estrutural:** ALTO (Potencial de danos catastróficos em áreas povoadas)
            *   **Ação M2M de AIoT:** Sinais automáticos de interrupção operacional transmitidos para infraestruturas críticas regionais (Metrô, Distribuidoras de Gás e Energia).
            """)
        elif mag_final >= 3.5:
            st.warning(f"""
            ### ⚠️ STATUS: MONITORAMENTO ATIVO / ATENÇÃO
            *   **Magnitude Estimada:** {mag_final:.2f} Mw
            *   **Risco Estrutural:** BAIXO (Tremor de terra perceptível na superfície)
            *   **Ação M2M de AIoT:** Logs de engenharia encaminhados para relatórios preventivos. Disparo de boletim informativo para aplicativos de defesa civil local.
            """)
        else:
            st.success(f"""
            ### 🟢 STATUS: OPERAÇÃO NORMAL
            *   **Magnitude Estimada:** {mag_final:.2f} Mw
            *   **Risco Estrutural:** INSIGNIFICANTE (Ruído de acomodação de placas estável)
            *   **Ação M2M de AIoT:** Apenas gravação de telemetria de rotina no banco de dados distribuído.
            """)
            
        fig_r, ax_r = plt.subplots(figsize=(6, 1.2))
        cor_barra = '#e74c3c' if mag_final >= 5.0 else ('#f1c40f' if mag_final >= 3.5 else '#2ecc71')
        ax_r.barh(['Risco'], [mag_final], color=cor_barra)
        ax_r.set_xlim(0, 9)
        ax_r.axvline(3.5, color='orange', linestyle='--')
        ax_r.axvline(5.0, color='red', linestyle='--')
        ax_r.set_xlabel("Escala Richter Estimada")
        st.pyplot(fig_r)
    else:
        st.info("💡 Carregue os parâmetros na barra lateral ou modifique o formulário para rodar a inferência da IA.")

with col2:
    st.subheader("📊 Avaliação Geral do Histórico de Desempenho")
    st.caption("Métricas consolidadas calculadas sobre os dados de teste reservados do USGS.")
    
    st.write("**Métricas do Trabalho Técnico:**")
    st.dataframe(df_metrics.style.format("{:.4f}"), use_container_width=True)
    
    st.markdown("---")
    st.write("**Gráficos Comparativos de Desempenho:**")
    
    # Gráfico RMSE
    fig_rmse, ax_rmse = plt.subplots(figsize=(6, 2))
    sns.barplot(x=df_metrics.columns, y=df_metrics.loc['RMSE (Regressão)'], ax=ax_rmse, palette="coolwarm")
    ax_rmse.set_title("Métrica de Regressão: RMSE (Menor é melhor)", fontsize=9, fontweight='bold')
    ax_rmse.set_ylabel("Erro (Mw)")
    for p in ax_rmse.patches:
        ax_rmse.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height() * 0.4),
                         ha='center', va='center', color='white', fontweight='bold', fontsize=8)
    st.pyplot(fig_rmse)
    
    # Gráficos Acurácia, Precisão, Recall, F1
    fig_class, axes = plt.subplots(1, 4, figsize=(10, 2.5))
    metrics_class = ["Acurácia", "Precisão", "Recall", "F1-Score"]
    
    for idx, metric_name in enumerate(metrics_class):
        ax = axes[idx]
        sns.barplot(x=df_metrics.columns, y=df_metrics.loc[metric_name], ax=ax, palette="viridis")
        ax.set_title(metric_name, fontsize=9, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("")
        
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height() * 0.5),
                             ha='center', va='center', color='white', fontweight='bold', fontsize=8)
            
    plt.tight_layout()
    st.pyplot(fig_class)


# BASE ACADÊMICA
st.markdown("---")
st.subheader("💡 Fundamentação Teórica das Variáveis USGS")
col_info1, col_info2 = st.columns(2)

with col_info1:
    st.info("""
    **Significado Técnico das Variáveis Utilizadas:**
    *   **rms (Root Mean Square):** Erro residual do tempo de viagem das ondas sísmicas (em segundos). Mede a precisão da leitura física do hardware IoT.
    *   **gap (Azimuthal Gap):** O maior ângulo cego de cobertura de estações ao redor do terremoto. Quanto menor o gap, mais cercado por sensores o epicentro estava.
    *   **sig (Significance):** Fator ponderado gerado por algoritmos do USGS para medir o quão relevante e impactante estruturalmente foi o tremor.
    """)

with col_info2:
    st.info("""
    **Justificativa de AIoT e Redes de Sensores:**
    Em aplicações de missão crítica (como defesa civil), a arquitetura híbrida de IA é fundamental. A **Regressão Linear** atua como um validador rápido na ponta de aquisição (*Edge Computing*), consumindo pouca energia, enquanto o **Random Forest** consolida os dados na camada de nuvem (*Cloud*) para acionar as respostas automatizadas máquina-para-máquina (M2M) de mitigação de danos.
    """)