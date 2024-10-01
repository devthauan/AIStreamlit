import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st

def plot_ips(df_tested_data, start_date='2022-08-13', end_date='2022-08-13'):
    # Filter the dataframe based on start and end date
    df_tested_data = df_tested_data[start_date:end_date]

    # Create a subplot with shared x-axis and two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Plot for Pressure (blue line on primary y-axis)
    fig.add_trace(
        go.Scatter(x=df_tested_data.index, y=df_tested_data['pressure(psig)'], 
                   mode='lines', name='Pressure (psig)', line=dict(color='blue', width=2)),
        secondary_y=False,
    )

    # Plot for Anomaly Probability (red dashed line on secondary y-axis)
    fig.add_trace(
        go.Scatter(x=df_tested_data.index, y=df_tested_data['anomaly_probability'], 
                   mode='lines', name='Anomaly Probability (%)', 
                   line=dict(color='red', width=1, dash='dash')),
        secondary_y=True,
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Date")

    # Set y-axes titles
    fig.update_yaxes(title_text="Pressure (psig)", secondary_y=False, color='blue')
    fig.update_yaxes(title_text="Anomaly Probability (%)", secondary_y=True, color='red')

    # Update the layout for the title and overall appearance
    fig.update_layout(
        title_text="Pressure vs Anomaly Probability",
        width=900,
        height=600,
        legend=dict(x=0.1, y=1.1, orientation='h')
    )

    # Add grid for better readability
    fig.update_xaxes(showgrid=True, gridwidth=1, )
    fig.update_yaxes(showgrid=False, gridwidth=1, )
    return fig

import plotly.express as px

def plot_anomaly_probability_histogram(df, n=100):
    """
    Plots a histogram of the distribution of the last n anomaly probabilities.

    Parameters:
    df (pd.DataFrame): DataFrame containing the 'anomaly_probability' column.
    n (int): The number of latest anomaly probabilities to include in the plot.
    """
    # Ensure 'anomaly_probability' exists in the DataFrame
    if 'anomaly_probability' not in df.columns:
        raise ValueError("'anomaly_probability' column is not found in the DataFrame")
    
    # Get the last n anomaly probabilities
    last_n_values = df['anomaly_probability'].tail(n)
    
    # Create the histogram using plotly
    fig = px.histogram(last_n_values, 
                       x="anomaly_probability", 
                       nbins=20, 
                       title=f"Distribution of Last {n} Anomaly Probabilities",
                       labels={'value':'Anomaly Probability'})
    
    # Customize the layout for better visualization
    fig.update_layout(
        xaxis_title='Anomaly Probability',
        yaxis_title='Count',
        bargap=0.2, # gap between bars
        showlegend=False
    )
    
    # Show the plot
    return fig

# Example usage:
# plot_anomaly_probability_histogram(df, n=100)

def anomaly_metrics(df, threshold=50,time_window='200000min'):

    df_last_minutes = df.last(time_window)
    anomalies = df_last_minutes['anomaly_probability']>=threshold
    total_anomalies = int(anomalies.sum())
    anomaly_percentage = (total_anomalies/len(df_last_minutes))*100
    anomaly_average_score = df_last_minutes.loc[anomalies, 'anomaly_probability'].mean()
    anomaly_std = df_last_minutes.loc[anomalies, 'anomaly_probability'].std()    
    max_anomaly = df_last_minutes['anomaly_probability'].max()

    stats = {
        'total_anomalies': total_anomalies,
        'anomaly_percentage': round(anomaly_percentage, 2),
        'avg_anomaly_score': round(anomaly_average_score, 2) if not pd.isna(anomaly_average_score) else np.nan,
        'anomaly_std': round(anomaly_std, 2) if not pd.isna(anomaly_std) else np.nan,
        'max_anomaly': round(max_anomaly,2)
    }

    return stats



def anomaly_visualization(data):
    # Título da aplicação
    st.title("Anomaly Probability Visualization")

    # Colocar a caixa de seleção na barra lateral
    st.sidebar.header("Choose the variable:")
    columns = ['pressure(psig)', 'bfw_turbine_speed(rpm)', 'bfw_pump_flow(gpm)']
    selected_column = st.sidebar.selectbox("Choose the column:", columns)

    # Gráfico da série temporal com a probabilidade de anomalia
    fig = go.Figure()

    # Adicionando a série temporal escolhida
    fig.add_trace(go.Scatter(x=data.index, y=data[selected_column],
                            mode='lines', name=selected_column))

    # Adicionando a probabilidade de anomalia
    fig.add_trace(go.Scatter(x=data.index, y=data['anomaly_probability'],
                            mode='lines', name='Anomaly Probability', line=dict(dash='dash')))

    # Configurações do layout
    fig.update_layout(title=f'{selected_column} vs Anomaly Probability',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    legend=dict(x=0, y=1, traceorder='normal'))
    return fig