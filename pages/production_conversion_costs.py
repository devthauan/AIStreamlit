import streamlit as st
import pandas as pd
import json
import urllib.request
from MyLibrary import ProductionConversionCosts
import ast
from MyLibrary import plot_costs


    


st.set_page_config(
    page_icon=":bar_chart:",
    page_title="Production Conversion Costs",
    )

st.title("Production Conversion Costs")

results_df = pd.read_csv( 'kpis_results.csv')
figure = plot_costs(results_df)
st.pyplot(figure)

@st.cache_data
def load_data(file):
    data = pd.read_excel(file)
    return data

uploaded_file = st.file_uploader("Select your excel file")

if uploaded_file is None:
    st.info("Upload a file", icon="⚠️")
    st.stop()

data = load_data(uploaded_file)
with st.expander("Data Preview"):
    st.dataframe(data)

answer = None
if st.button(label="Generate Optimization"):
    input_data = json.dumps({"data": json.loads(data.to_json()), "target_date": uploaded_file.name.split("_")[0] })
    api = ProductionConversionCosts()
    answer = api.call(input_data)

if(answer):
    json_answer = json.loads(answer)
    df_answer = pd.DataFrame.from_dict(ast.literal_eval(json_answer))
    st.dataframe(df_answer)