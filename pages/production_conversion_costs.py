import streamlit as st
import pandas as pd
import json
import urllib.request
from MyLibrary import ProductionConversionCosts
import ast
from MyLibrary import plot_costs, plot_costs_plotly, plot_reduction_plotly


st.set_page_config(
    page_icon=":bar_chart:",
    page_title="Production Conversion Costs",
    )

st.title("Production Conversion Costs")

# Create a container for the first part
with st.container():
    st.header("Efficiency Metrics", divider=True)

    # Create a container to arrange buttons horizontally
    col1, col2, col3 = st.columns(3)
    # Place buttons within the columns
    with col1:
        button1 = st.button("Actual x Optimized Cost",  use_container_width=True)
    with col2:
        button2 = st.button("Cost Reduction %",  use_container_width=True)
    with col3:
        button3 = st.button("TimeSpent Reduction %",  use_container_width=True)

    # Set the session state to indicate the button was clicked
    st.session_state['button_clicked'] = True

    # Check if the button was clicked
    if (button1 or st.session_state.get('button_clicked', False)) and (not(button2) and not(button3)):
    #if button1:
        results_df = pd.read_csv( 'kpis_results.csv')
        figure = plot_costs_plotly(results_df)
        st.plotly_chart(figure)
    if button2:
        results_df = pd.read_csv( 'kpis_results.csv')
        figure = plot_reduction_plotly(results_df, change="Cost", unit="%", pad_max=10, pad_min=3)
        st.plotly_chart(figure)
    if button3:
        results_df = pd.read_csv( 'kpis_results.csv')
        figure = plot_reduction_plotly(results_df, change="TimeSpent", unit="%", pad_max=10, pad_min=3)
        st.plotly_chart(figure)

# Create a container for the second part
with st.container():
    st.header("Plan Optimal Allocation", divider=True)

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