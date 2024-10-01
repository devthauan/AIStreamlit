import streamlit as st
import pandas as pd
from library import plot_ips
from datetime import datetime
import pytz
from CRONJOB_incident_probability_score import retrieve_cognite_data,retrieve_model_cognite_data
import plotly.io as pio
from library import plot_anomaly_probability_histogram, anomaly_metrics, anomaly_visualization

st.set_page_config(
    layout="wide",
    page_icon=":bar_chart:",
    page_title="Production Conversion Costs",
    )

tzinfo= pytz.utc
start_time = pd.to_datetime('2024-06-08 00:00:00')
end_time = datetime.now(tzinfo) 

df_extract = retrieve_cognite_data(start_time=start_time, end_time=end_time)
df_model_extract = retrieve_model_cognite_data(start_time=start_time, end_time=end_time)
df_model_extract.columns = ['anomaly_probability']
df = df_extract.merge(df_model_extract, on='timestamp')
df["anomaly_probability"] = df["anomaly_probability"].clip(upper=100, lower=0)

# Creating a subheader
st.subheader("Incident Probability Score - Pump P 1221")

# Set the default template globally
pio.templates.default = "ggplot2" 

anomaly_stats = anomaly_metrics(df)
anomaly_col1, anomaly_col2, anomaly_col3, anomaly_col4, anomaly_col5 = st.columns(5)
with anomaly_col1:
    st.metric(label = "Total Anomalies", value = anomaly_stats["total_anomalies"])
with anomaly_col2:
    st.metric(label = "Anomaly Percentage", value = anomaly_stats["anomaly_percentage"])
with anomaly_col3:
    st.metric(label = "Average Anomaly Score", value = anomaly_stats["avg_anomaly_score"])
with anomaly_col4:
    st.metric(label = "Anomaly Standand Deviation", value = anomaly_stats["anomaly_std"])
with anomaly_col5:
    st.metric(label = "Maximum Probability", value = anomaly_stats["max_anomaly"])

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(plot_ips(df, start_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d') ), theme= None )
with col2:
    st.plotly_chart(plot_anomaly_probability_histogram(df, n=100000), theme= None)

# Exibir o gráfico
# anomaly_visualization = anomaly_visualization(df)
# st.plotly_chart(anomaly_visualization)