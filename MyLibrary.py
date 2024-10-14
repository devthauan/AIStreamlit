import streamlit as st
import time
import numpy as np
import pandas as pd 
from typing import Dict
import json, urllib
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

class AIMVP:
    def __init__(self, url: str, api_key: str):
        self.api_key = api_key
        self.url = url

    def call(self, data):
        body = self.prepare_payload(data)
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ self.api_key)}
        req = urllib.request.Request(self.url, body, headers)

        try:
            response = urllib.request.urlopen(req)
            return self.prepare_output(response.read())
        except urllib.error.HTTPError as error:
            print("The request failed with status code: " + str(error.code))

    def prepare_payload(self, data) -> Dict:
        raise NotImplementedError

    def prepare_output(self, data):
        raise NotImplementedError


class RootCauseAnalysis(AIMVP):
    def __init__(self):
        super().__init__('https://rca-endpoint.eastus.inference.ml.azure.com/score', st.secrets["RCA_API_KEY"])

    def prepare_payload(self, data) -> Dict:
        return str.encode(json.dumps({"question": data}))

    def prepare_output(self, data):
        return data.decode('utf-8')

class ProductionConversionCosts(AIMVP):
    def __init__(self):
        super().__init__('https://production-conversion-cost.eastus.inference.ml.azure.com/score', st.secrets["PCC_API_KEY"])

    def prepare_payload(self, data) -> Dict:
        return str.encode(data)

    def prepare_output(self, data):
        return data.decode('utf-8')


def plot_costs(results_df):
    figure = plt.figure(figsize=(15, 6))
    results_df = results_df.sort_values("Date")
    plt.plot(results_df["Date"], results_df["ActualCost"],  'o-', label="Actual Cost")
    plt.plot(results_df["Date"], results_df["OptimizedPredCost"],  'o-', label="Model Optimized Cost")
    plt.title("Actual Cost x Model Optimized Cost")
    plt.ylabel("Cost ($)")
    plt.xlabel("Date")
    plt.xticks(rotation=15)
    plt.legend()
    return figure

def plot_costs_plotly(results_df):

    results_df = results_df.sort_values(by="Date")

    plot_layout = dict( margin=dict(l=48, r=48, t=16, b=32),
                        xaxis=dict(showgrid=True, zeroline=True, gridcolor="rgba(255, 255, 255, .6)"),  
                        yaxis=dict(showgrid=True, zeroline=True, gridcolor="rgba(255, 255, 255, .6)"),  
                        # plot_bgcolor="rgba(232, 232, 232, 1)",#plot_bgcolor="rgba(0, 0, 0, 0)",    
                        # paper_bgcolor="rgba(232, 232, 232, 1)",  #paper_bgcolor="rgba(0,0,0,0)",    
                        # font_family="DIN Alternate",    
                        showlegend=True,    
                        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),)
        
    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = results_df["Date"], y=results_df["ActualCost"],
                        mode='lines+markers',
                        name='Actual Cost'))
    fig.add_trace(go.Scatter( x=results_df["Date"], y=results_df["OptimizedPredCost"],
                        mode='lines+markers',
                        name='Model Optimized Cost'))
    # Edit the layout
    fig.update_layout(title='Actual Cost x Model Optimized Cost',
                   xaxis_title='Cost ($)',
                   yaxis_title='Date')
    fig.update_layout(**plot_layout)

    return fig

def plot_reduction_plotly(results_df, change="Cost", unit="%", pad_max=10, pad_min=3):
    
    plot_layout = dict( margin=dict(l=48, r=48, t=16, b=32),
                        xaxis=dict(showgrid=True, zeroline=True,gridcolor="rgba(255, 255, 255, .6)"),  
                        yaxis=dict(showgrid=True, zeroline=True, gridcolor="rgba(255, 255, 255, .6)"),  
                        # plot_bgcolor="rgba(232, 232, 232, 1)",#plot_bgcolor="rgba(0, 0, 0, 0)",    
                        # paper_bgcolor="rgba(232, 232, 232, 1)",  #paper_bgcolor="rgba(0,0,0,0)",    
                        # font_family="DIN Alternate",    
                        showlegend=True,    
                        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),)

    results_df["Date"] = pd.to_datetime(results_df["Date"], format='%Y-%m-%d')
    results_df = results_df.set_index("Date")
    
    results_df["Feasibility"] = ""
    results_df.loc[results_df['Status'] == 1, 'Feasibility'] = "feasible" 
    results_df.loc[results_df['Status'] != 1, 'Feasibility'] = "unfeasible"
    fig = px.bar(results_df, x=results_df.index, y=results_df[f"Change{change}"], color=results_df['Feasibility'],
                 color_discrete_map={   'feasible' : 'orange',
                                        'unfeasible' : 'red'})
    
    # Edit the layout
    fig.update_layout(title=f"{unit} {change} reduction by Optimization model per month",
                   xaxis_title='Date',
                   yaxis_title=f"{change} reduction ({unit})")
    fig.update_layout(**plot_layout)
    
    return fig

def plot_allocation_plotly(df_pred, month, setup_time=0.08):
    pred = df_pred
    periods = {1: 31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    pred = pred[pred["RunTimeHrs"] > 0]

    # Add setup
    setups = []
    for row in pred.itertuples():    
        setups.append({ "Material": f"{row.Material} Setup",
                        "Line": row.Line,
                        "RunTimeHrs": setup_time })    
    pred2 = pd.concat([pred, pd.DataFrame(setups)])
    # Add slack
    slack = pred2.groupby("Line", as_index=False).agg({"RunTimeHrs": "sum"})
    slack['RunTimeHrs'] = (periods[month] * 24 - slack['RunTimeHrs']).round(2).apply(lambda x:x if abs(x) > 0.01 else 0)
    slack['Material'] = "Slack"
    # Convert material to str and sort names
    pred2 = pred2.astype({"Material": str})
    pred2 = pred2.sort_values(by=["Material", "Line"], ascending=False)
    # Concat pred data to slack
    pred2 = pd.concat([pred2, slack])

    plot_layout = dict( margin=dict(l=48, r=48, t=48, b=48),
                    xaxis=dict(showgrid=False, zeroline=False, gridcolor="rgba(0, 0, 0, 0)"),   
                    yaxis=dict(showgrid=False, zeroline=False, gridcolor="rgba(0, 0, 0, 0)"),   
                    # plot_bgcolor="rgba(232, 232, 232, 1)",#plot_bgcolor="rgba(0, 0, 0, 0)",    
                    # paper_bgcolor="rgba(232, 232, 232, 1)",  #paper_bgcolor="rgba(0,0,0,0)",    
                    # font_family="DIN Alternate",    
                    showlegend=True,    
                    )

    fig = px.bar(pred2.astype({"Material": str}), y="Line", x="RunTimeHrs", 
                color="Material", text="Material", height=600)

    fig.update_traces(textposition='inside')
    fig.update_layout(uniformtext_minsize=10, 
                    uniformtext_mode='hide',
                    title="Optimized Allocation",
                    xaxis_title="Time",
                    yaxis_title="Line",
                    )

    fig.update_layout(**plot_layout)

    for trace in fig.data:
    # Set layer color as gray when the time spent is due to setup or slack
        if "Setup" in trace.name or "Slack" in trace.name:
            trace.marker.color = 'gray'
            trace.showlegend = False

    return fig