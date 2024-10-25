import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import os, sys, logging, pytz
from datetime import datetime, timedelta, time
import datetime as dt
# from datetime import datetime ,timedelta
from joblib import load

st.set_page_config(
    layout="wide",
    page_icon=":bar_chart:",
    page_title="Incident Probability Score",
    )
# Reading data from cognite
from extracting_process_timeseries import ExtractProcessTimeseries
from miscellaneous import merge_to_streamlit
# MÉTRICAS DA ANOMALIA ################
@st.cache_data(ttl = '5min')
def get_data_from_cognite(env):
    ips_path = 'boilers_dashboard.json'

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
    tzinfo = pytz.timezone('America/Chicago')
    end_time = datetime.now(tzinfo)
    start_time = end_time - timedelta(days = 180)
    df = read_write.get_timeseries_data(start_time = start_time, end_time = end_time)    
    return df

df_outputs = get_data_from_cognite("prod")
df_tested_data = df_outputs.copy().bfill()

def anomaly_metrics(df, boiler_anomaly_percent, threshold=50, start_time=None, end_time=None):

    df_boiler_anomalies = df[(df.index>=start_time) & (df.index<=end_time)]
    anomalies = df_boiler_anomalies[boiler_anomaly_percent] >= threshold
    total_anomalies = anomalies.sum()
    anomaly_percentage = (total_anomalies / len(df_boiler_anomalies)) * 100
    anomaly_average_score = df_boiler_anomalies.loc[:, boiler_anomaly_percent].mean()
    anomaly_std = df_boiler_anomalies.loc[:, boiler_anomaly_percent].std()
    max_anomaly = df_boiler_anomalies[boiler_anomaly_percent].max()

    stats = {
        'total_anomalies': int(total_anomalies),
        'anomaly_percentage': round(anomaly_percentage, 2),
        'avg_anomaly_score': round(anomaly_average_score, 2) if not pd.isna(anomaly_average_score) else np.nan,
        'anomaly_std': round(anomaly_std, 2) if not pd.isna(anomaly_std) else np.nan,
        'max_anomaly': round(max_anomaly,2)
    }
    return stats

################# DASHBOARD NAME #################
st.title("Anomaly Probability Score Dashboard: Boilers")


################# SETTING DATETIME RANGE #################

date_max = df_tested_data.index.max()

# select default for start date, end date and end time
# start_date = st.sidebar.date_input("Start Date", value=date_max - timedelta(days = 7))
# start_datetime_metrics = pd.to_datetime(f"{start_date} {df_tested_data[start_date:start_date].index.min().time()}")
# end_date = st.sidebar.date_input("End Date", value=date_max)

tzinfo = pytz.utc
# date_max = df_tested_data.index.max()
today = dt.date.today()
end_time = datetime.now(tzinfo)
start_time = (end_time - timedelta(days = 180)).date()
start_date = st.sidebar.date_input("Start Date", value=today - timedelta(days = 7), 
                                   min_value = start_time, max_value = today-timedelta(days = 1))
end_date = st.sidebar.date_input("End Date", value=today, min_value = start_date, max_value=today)

if end_date > today or start_date <= start_time:
    st.markdown("<h3 style='color: red;'>You cannot select a previous date before six months or a future date.</h3>", unsafe_allow_html=True)
else:
    st.markdown(f"<h4 style='color: blue;'>Showing data up until the current date: {end_date}</h4>", unsafe_allow_html=True)

    # chose interval time by user (max of 24hs)
    end_time = st.sidebar.time_input("Select the End Time (related to end date)", 
                                    value=df_tested_data.index.max().time())
    time_window_hours = st.sidebar.slider("Select the Time Window (in hours)", 0, 24, 1)

    # datetime interval for the calculate of metrics (big numbers view)
    start_datetime_metrics = pd.to_datetime(f"{start_date} {df_tested_data[start_date:start_date].index.min().time()}")

    # datetime interval for the last hours in the time series
    end_datetime = pd.to_datetime(f"{end_date} {end_time}")
    start_datetime = end_datetime - pd.Timedelta(hours=time_window_hours)

    # Filtrando o DataFrame com base no intervalo de datetime selecionado

    filtered_data = df_tested_data[start_datetime:end_datetime]
    filtered_data_eff = filtered_data[[col for col in filtered_data if "efficiency" in col.lower()]].dropna()
    filtered_data_anom = filtered_data[[col for col in filtered_data if "anomaly" in col.lower()]]

    if len(filtered_data)==0:
        st.warning("There are no data in this period")

    # select the boiler
    selected_column = st.sidebar.selectbox(
        'Choose Boiler',
        ('MB6', 'MB120', 'MB130', 'MB140', 'MB150')
    )
    # select the column based on the boiler selection
    boiler_efficieny_column = f'{selected_column}_Efficiency'
    boiler_anomaly_percent = f'Anomaly_{selected_column}'


    ################# ANOMALY METRICS #################
    anomaly_stats = anomaly_metrics(df_tested_data, boiler_anomaly_percent, start_time=start_datetime_metrics, end_time=end_datetime)
    st.markdown(f'<span style="font-size: 20px;">**Anomaly Metrics From the date interval: {start_date} to {end_date}**</span>', unsafe_allow_html=True)

    # display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Anomalies", anomaly_stats['total_anomalies'])
    col2.metric("Anomaly %", f"{anomaly_stats['anomaly_percentage']}%")
    col3.metric("Avg Anomaly Score", f"{anomaly_stats['avg_anomaly_score']}%")
    col4.metric("Max Anomaly", f"{anomaly_stats['max_anomaly']}%")
    col5.metric("Std Anomaly", anomaly_stats['anomaly_std'])

    ################# PLOT TIME SERIES X ANOMALY PERCENTAGE (GRAPHIC 1,1) #################

    df_date_interval = df_tested_data[start_date:end_date]
    fig = go.Figure()

    # adding time series (for range in start date and end date)
    # efficiency of the boiler
    fig.add_trace(go.Scatter(x=df_date_interval.index, y=df_date_interval[boiler_efficieny_column],
                            mode='lines', name=selected_column, line=dict(color='blue')))
    # anomaly of the boiler
    fig.add_trace(go.Scatter(x=df_date_interval.index, y=df_date_interval[boiler_anomaly_percent],
                            mode='lines', name='anomaly(%)',
                            line=dict(color='orange', dash='dash'),
                            yaxis='y2'))

    fig.update_layout(
        title=f'{boiler_efficieny_column} vs {boiler_anomaly_percent} {start_date} to {end_date}',
        xaxis_title='Date',
        yaxis_title=boiler_efficieny_column,
        yaxis1 = dict(range=[0,100]),
        yaxis2=dict(
            title="Anomaly Probability (%)",
            overlaying='y',
            side='right',
            range=[0,100]
        ),
        legend=dict(x=0, y=1, traceorder='normal')
    )

    ################# PLOT TIME SERIES X ANOMALY PERCENTAGE FOR LAST HOURS (GRAPHIC 1,2) #################

    fig_last_x_min = go.Figure()

    # adding time series (for range in the last hours for the end date)
    # efficiency of the boiler
    fig_last_x_min.add_trace(go.Scatter(x=filtered_data_eff.index, y=filtered_data_eff[boiler_efficieny_column],
                                        mode='lines', name=selected_column, line=dict(color='blue')))
    # anomaly of the boiler
    fig_last_x_min.add_trace(go.Scatter(x=filtered_data_anom.index, y=filtered_data_anom[boiler_anomaly_percent],
                                        mode='lines', name='anomaly(%)',
                                        line=dict(color='orange', dash='dash'),
                                        yaxis='y2'))

    fig_last_x_min.update_layout(
        title=f'{boiler_efficieny_column} (last {time_window_hours} hours related to End Date and End Time) vs Anomaly',
        xaxis_title='Date',
        yaxis_title=boiler_efficieny_column,
        yaxis1 = dict(range=[0,100]),
        yaxis2=dict(
            title="Anomaly Probability (%)",
            overlaying='y',
            side='right',
            range=[0,100]
        ),
        legend=dict(x=0, y=1, traceorder='normal')
    )

    ################# PLOT TIME HISTOGRAM WITH BOXPLOT (GRAPHIC 2,1) #################

    fig_hist_box = px.histogram(df_date_interval, x=boiler_anomaly_percent, nbins=30, marginal="box", color_discrete_sequence=['green'])
    fig_hist_box.update_layout(
        title=f"Anomaly Probability Distribution {start_date} to {end_date}",
        xaxis_title="Anomaly Probability",
        yaxis_title="Count",
        showlegend=False
    )

    ################# PLOT TIME STACK BARS (GRAPHIC 2,2) #################

    # setting the last 30 days
    first_date = end_datetime - pd.Timedelta(days=30)
    df_date_interval = df_tested_data[first_date:end_datetime]

    # classifying the anomalies and no anomalies
    df_date_interval['anomaly'] = (df_date_interval[boiler_anomaly_percent] >= 50).astype(int)
    df_date_interval['no_anomaly'] = (df_date_interval[boiler_anomaly_percent] < 50).astype(int)

    # resamples 'anomaly' and 'no_anomaly' columns daily, summing and filling NaNs with 0.
    df_bars = df_date_interval[['anomaly', 'no_anomaly']].resample('1d').sum().fillna(0)
    df_bars.columns = ['Anomalies', 'Inliers']

    # calculate percentage of anomalies and not anomalies
    df_bars['Total'] = df_bars['Anomalies'] + df_bars['Inliers']
    df_bars['Anomalies_percent'] = (df_bars['Anomalies']/df_bars['Total'])*100
    df_bars['Inliers_percent'] = (df_bars['Inliers']/df_bars['Total'])*100

    # setting stacked bars
    fig_bars = go.Figure()
    fig_bars.add_trace(go.Bar(x=df_bars.index, y=df_bars['Anomalies_percent'], name='Anomalies', marker_color='red'))
    fig_bars.add_trace(go.Bar(x=df_bars.index, y=df_bars['Inliers_percent'], name='Inliers', marker_color='blue'))

    fig_bars.update_layout(
        title=f'Anomaly and Inlier Percentages by Day of the Week {first_date.date()} to {end_datetime.date()}',
        xaxis_title='Day of the Week',
        yaxis_title='Percentage',
        barmode='stack',
        yaxis=dict(range=[0, 100], tickformat='.0f'),
        legend=dict(x=0, y=1)
    )

    ################# SETTING DASBOARD VIEW #################

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.plotly_chart(fig_last_x_min, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_hist_box, use_container_width=True)

    with col2:
        st.plotly_chart(fig_bars, use_container_width=True)