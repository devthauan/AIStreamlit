import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import  pytz
from datetime import datetime ,timedelta

st.set_page_config(
    layout="wide",
    page_icon=":bar_chart:",
    page_title="Incident Probability Score",
    )
# MÉTRICAS DA ANOMALIA ################
def anomaly_metrics(df, threshold=50, time_window='20min'):
    df_last_minutes = df.last(time_window)
    anomalies = df_last_minutes['anomaly_probability'] >= threshold
    total_anomalies = anomalies.sum()
    anomaly_percentage = (total_anomalies / len(df_last_minutes)) * 100
    anomaly_average_score = df_last_minutes.loc[anomalies, 'anomaly_probability'].mean()
    anomaly_std = df_last_minutes.loc[anomalies, 'anomaly_probability'].std()
    max_anomaly = df_last_minutes['anomaly_probability'].max()

    stats = {
        'total_anomalies': total_anomalies,
        'anomaly_percentage': round(anomaly_percentage, 2),
        'avg_anomaly_score': round(anomaly_average_score, 2) if not pd.isna(anomaly_average_score) else np.nan,
        'anomaly_std': round(anomaly_std, 2) if not pd.isna(anomaly_std) else np.nan,
        'max_anomaly': round(max_anomaly, 2)
    }
    return stats

# TÍTULO DO DASHBOARD ################
st.title("Anomaly Probability Score Dashboard")
# Reading data from cognite
from extracting_process_timeseries import ExtractProcessTimeseries
from miscellaneous import merge_to_streamlit

@st.cache_data
def get_data_from_cognite(env):
    ips_path = 'probability_score.json'

    read_write = ExtractProcessTimeseries(
            path = ips_path, 
            client_name = st.secrets["CLIENT_NAME"],
            base_url = st.secrets["BASE_URL"],
            authority = st.secrets["AUTHORITY"],
            client_id = st.secrets["CLIENT_ID"],
            scopes = st.secrets["SCOPES"],
            username = st.secrets["COGNITE_USERNAME"],
            password = st.secrets["COGNITE_PASSWORD"],
            env = env
    )
    tzinfo = pytz.utc
    end_time = datetime.now(tzinfo)
    start_time = end_time - timedelta(days = 180)
    df = read_write.get_timeseries_data(start_time = start_time, end_time = end_time)
    df_outputs = read_write.get_timeseries_data(start_time = start_time,end_time = end_time,in_out = 'output')
    df_outputs = merge_to_streamlit(df,df_outputs,ips_path,'P1221 Incident Prediction Probability')
    df_outputs = df_outputs[df_outputs["anomaly_probability"].notnull()]
    df_outputs["anomaly_probability"] = df_outputs["anomaly_probability"].clip(upper=100, lower=0)
    return df_outputs


df_outputs = get_data_from_cognite("dev")

# LEITURA DO ARQUIVO ################
df_tested_data = df_outputs

if not pd.api.types.is_datetime64_any_dtype(df_tested_data.index):
    df_tested_data.index = pd.to_datetime(df_tested_data.index)

# CAIXA NA BARRA LATERAL ################
st.sidebar.header("Variables")
columns = ['Pressure', 'Speed', 'Flow']
selected_column = st.sidebar.selectbox("Choose the time series:", columns)

# CALCULO DAS METRICAS ################
anomaly_stats = anomaly_metrics(df_tested_data)

# CALCULO DO DESVIO PADRAO PLOT(1,1) ################
# df_tested_data['rolling_std'] = df_tested_data['anomaly_probability'].rolling('60min').std()

# DISPLAY BIGNUMBERS ################
# st.markdown("### Anomaly Metrics Last 20 minutes")
st.markdown('<span style="font-size: 18px;">**Anomaly Metrics Last 20 minutes**</span>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Anomalies", anomaly_stats['total_anomalies'])
col2.metric("Anomaly %", f"{anomaly_stats['anomaly_percentage']}%")
col3.metric("Avg Anomaly Score", f"{anomaly_stats['avg_anomaly_score']}%")
col4.metric("Max Anomaly", f"{anomaly_stats['max_anomaly']}%")
col5.metric("Std Anomaly", anomaly_stats['anomaly_std'])

# PLOT SÉRIE TEMPORAL X ANOMALIA (1,1) ################
fig = go.Figure()

# Adicionando a série temporal escolhida
fig.add_trace(go.Scatter(x=df_tested_data.index, y=df_tested_data[selected_column],
                         mode='lines', name=selected_column, line=dict(color='blue')))

fig.add_trace(go.Scatter(x=df_tested_data.index, y=df_tested_data['anomaly_probability'],
                         mode='lines', name='anomaly(%)',
                         line=dict(color='orange', dash='dash'),
                         yaxis='y2'))
# Configurações do layout com y-axis secundário
fig.update_layout(
    title=f'{selected_column} vs Anomaly Probability',
    xaxis_title='Date',
    yaxis_title=selected_column,
    yaxis2=dict(
        title="Anomaly Probability (%)",
        overlaying='y',
        side='right'
    ),
    legend=dict(x=0, y=1, traceorder='normal')
)

# Calcular o rolling_std com uma janela de 5 minutos para os últimos 100 minutos
df_last_100_min = df_tested_data.last('100min')
# df_last_100_min['rolling_std_5min'] = df_last_100_min['anomaly_probability'].rolling('5min').std()

# PLOT SÉRIE TEMPORAL X ANOMALIA (1,2) ################
fig_last_100 = go.Figure()

# Adicionando a série temporal escolhida
fig_last_100.add_trace(go.Scatter(x=df_last_100_min.index, y=df_last_100_min[selected_column],
                                    mode='lines', name=selected_column, line=dict(color='blue')))

# Adicionando o desvio padrão no eixo y secundário (cor laranja)
fig_last_100.add_trace(go.Scatter(x=df_last_100_min.index, y=df_last_100_min['anomaly_probability'],
                                    mode='lines', name='anomaly(%)',
                                    line=dict(color='orange', dash='dash'),
                                    yaxis='y2'))

# Configurações do layout do gráfico de últimos 100 minutos
fig_last_100.update_layout(
    title=f'{selected_column} (Last 100 min) vs Anomaly Probability',
    xaxis_title='Date',
    yaxis_title=selected_column,
    yaxis2=dict(
        title="Anomaly Probability (%)",
        overlaying='y',
        side='right'
    ),
    legend=dict(x=0, y=1, traceorder='normal')
)

# PLOT HISTOGRAMA COM BLOXPOT (2,1) ################
fig_hist_box = px.histogram(df_tested_data, x='anomaly_probability', nbins=30, marginal="box", color_discrete_sequence=['green'])
fig_hist_box.update_layout(
    title="Anomaly Probability Distribution",
    xaxis_title="Anomaly Probability",
    yaxis_title="Count",
    showlegend=False
)

# PLOT BARRAS EMPILHADAS (2,2) ################

# Classificação de anomalia
df_tested_data['anomaly'] = (df_tested_data['anomaly_probability'] >= 50).astype(int)

# Adicionar colunas para o dia da semana
df_tested_data['weekday'] = df_tested_data.index.weekday  # 0 = Monday, 6 = Sunday

# Contar a quantidade de anomalias (1) e inliers (0) por dia da semana
anom_count = df_tested_data[df_tested_data['anomaly'] == 1].groupby('weekday').size()
non_anom_count = df_tested_data[df_tested_data['anomaly'] == 0].groupby('weekday').size()

# Criar um DataFrame para organizar os dados
df_bars = pd.DataFrame({
    'Anomalies': anom_count,
    'Inliers': non_anom_count
}).fillna(0)  # Preencher valores NaN com 0

all_days = pd.Index(range(7), name='weekday')
df_bars = df_bars.reindex(all_days, fill_value=0)

df_bars['Total'] = df_bars['Anomalies'] + df_bars['Inliers']
df_bars['Anomalies_percent'] = (df_bars['Anomalies']/df_bars['Total'])*100
df_bars['Inliers_percent'] = (df_bars['Inliers']/df_bars['Total'])*100

# Definir os rótulos dos dias da semana
days_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# df_bars = df_bars.reindex(range(7), fill_value=0)
# Gráfico de barras empilhadas
fig_bars = go.Figure()
fig_bars.add_trace(go.Bar(x=days_labels, y=df_bars['Inliers_percent'], name='Inliers', marker_color='blue'))
fig_bars.add_trace(go.Bar(x=days_labels, y=df_bars['Anomalies_percent'], name='Anomalies', marker_color='red'))

# Configurações do layout do gráfico de barras
fig_bars.update_layout(
    title='Anomaly and Inlier Percentages by Day of the Week',
    xaxis_title='Day of the Week',
    yaxis_title='Percentage',
    barmode='stack',
    yaxis=dict(range=[0, 100], tickformat='.0f'),
    legend=dict(x=0, y=1)
)

# Layout do painel: Primeira linha com dois gráficos lado a lado
# st.markdown("## Time Series Analysis")
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.plotly_chart(fig_last_100, use_container_width=True)

# Segunda linha com mais dois gráficos lado a lado
# st.markdown("## Anomaly Probability and Day of the Week Analysis")
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_hist_box, use_container_width=True)

with col2:
    st.plotly_chart(fig_bars, use_container_width=True)
