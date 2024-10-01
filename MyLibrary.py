import streamlit as st
import time
import numpy as np
from typing import Dict
import json, urllib
import os
import matplotlib.pyplot as plt

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
        super().__init__('https://pcc-endpoint.eastus.inference.ml.azure.com/score', st.secrets["PCC_API_KEY"])

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