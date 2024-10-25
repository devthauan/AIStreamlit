import streamlit as st
import re
import pandas as pd
import json
import urllib.request
from MyLibrary import ProductionConversionCosts, ProductionCostEstimator
import ast
from MyLibrary import plot_costs, plot_costs_plotly, plot_reduction_plotly, plot_allocation_plotly, plot_cost_by_line_material,format_large_number, plot_cost_by_material, plot_cost_by_machine
import numpy as np

def update_graphic_figure(figure):
    st.session_state.graphic_figure = figure

def update_graphic_info_text(value):
    st.session_state.graphic_info_text = value

st.set_page_config(
    layout="wide",
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
        button_graphic_atual_x_opt_cost = st.button("Actual x Optimized Cost",  use_container_width=True)
    with col2:
        button_graphic_cost_reduction = st.button("Cost Reduction %",  use_container_width=True)
    with col3:
        button_graphic_timespent_reduction = st.button("TimeSpent Reduction %",   use_container_width=True)
    # Set the session state to indicate the button was clicked
    st.session_state['button_clicked'] = True

    # Initialize session state
    if 'graphic_info_text' not in st.session_state:
        graphic_graphic_atual_x_opt_cost_info = "The plant has consistently operated at a higher cost than the optimized allocation. Over the 17-month period, the actual costs have exceeded the optimized costs by an average of 609,961 dollars per month, leading to a cumulative excess of 10.37 million dollars. This indicates that there is substantial room for cost reduction by aligning operations with the optimization model. The most significant deviation occurred in May 2024, where actual costs were $2.57 million higher than the optimized costs. This suggests a major inefficiency or operational issue during that month, with costs running 48% higher than the model's recommendations. On the positive side, there were two months where the plant operated below the optimized cost. October 2023 and January 2024 both showed cost savings, with January 2024 standing out as the most efficient month, where actual costs were 19% lower than predicted.The plant’s average efficiency ratio is 1.14, meaning actual costs have been about 14% higher than optimized costs on average. While some months, like June 2023 and October 2023, exhibited a favorable efficiency ratio, most months reflect inefficiency."
        st.session_state.graphic_info_text = graphic_graphic_atual_x_opt_cost_info
    # Initialize session state
    if 'graphic_figure' not in st.session_state:
        results_df = pd.read_csv( 'kpis_results.csv')
        figure = plot_costs_plotly(results_df)
        st.session_state.graphic_figure = figure

    # Check if the button was clicked
    #if (button1 or st.session_state.get('button_clicked', False)) and (not(button2) and not(button3)):
    if (button_graphic_atual_x_opt_cost):
        results_df = pd.read_csv( 'kpis_results.csv')
        figure = plot_costs_plotly(results_df)
        graphic_graphic_atual_x_opt_cost_info = "The plant has consistently operated at a higher cost than the optimized allocation. Over the 17-month period, the actual costs have exceeded the optimized costs by an average of 609,961 dollars per month, leading to a cumulative excess of 10.37 million dollars. This indicates that there is substantial room for cost reduction by aligning operations with the optimization model. The most significant deviation occurred in May 2024, where actual costs were $2.57 million higher than the optimized costs. This suggests a major inefficiency or operational issue during that month, with costs running 48% higher than the model's recommendations. On the positive side, there were two months where the plant operated below the optimized cost. October 2023 and January 2024 both showed cost savings, with January 2024 standing out as the most efficient month, where actual costs were 19% lower than predicted.The plant’s average efficiency ratio is 1.14, meaning actual costs have been about 14% higher than optimized costs on average. While some months, like June 2023 and October 2023, exhibited a favorable efficiency ratio, most months reflect inefficiency."
        update_graphic_info_text(graphic_graphic_atual_x_opt_cost_info)
        update_graphic_figure(figure)
    if button_graphic_cost_reduction:
        results_df = pd.read_csv( 'kpis_results.csv')
        figure = plot_reduction_plotly(results_df, change="Cost", unit="%", pad_max=10, pad_min=3)
        graphic_cost_reduction_info = "The optimization model has consistently led to cost savings in 14 out of 17 months, with reductions ranging from 4.1% to 32.4%. The highest cost reduction occurred in May 2024, where the model demonstrated a 32.44% reduction in estimated costs compared to actual allocations. This is a significant indication of the model’s potential to drive efficiency and cost reduction when applied correctly.There are three months where the optimization model estimated a negative cost reduction (an increase), suggesting that the actual allocation was more cost-effective. The most prominent examples are in October 2023 and January 2024, where costs increased by 9.28% and 23.13%, respectively, when following the optimization model. This may indicate anomalies in the data or specific circumstances where the plant's actual allocation was more efficient than expected.Besides May 2024, other months of notable efficiency gains include August 2023 and September 2023, with cost reductions of 24.82% and 29.39%, respectively. These months suggest a strong alignment between the optimization model's recommendations and potential cost savings."
        update_graphic_info_text(graphic_cost_reduction_info)
        update_graphic_figure(figure)
    if button_graphic_timespent_reduction:
        results_df = pd.read_csv( 'kpis_results.csv')
        figure = plot_reduction_plotly(results_df, change="TimeSpent", unit="%", pad_max=10, pad_min=3)
        graphic_timespent_reduction_info = "In 15 out of 17 months, the optimization model successfully reduced the time spent on production allocation, with reductions ranging from 2.99% to 40.05%. The largest time reduction occurred in April 2023, with a 40.05% decrease in time compared to the actual allocation, indicating that the optimization model greatly improved efficiency in time usage during that period.The model estimated negative reductions (increases in time) in two months: June 2023 and October 2023. The most significant time increase occurred in June 2023, where the time spent on production increased by 10.51% compared to the actual allocation. This suggests inefficiencies or model inaccuracies during these months, where the plant's actual allocation was more time-efficient than the optimized recommendation.From January 2024 to August 2024, the optimization model consistently provided time savings, with reductions ranging from 11.75% to 18.17%. This highlights a stable period of improvement, showing that the model consistently helped in reducing production time during this phase."
        update_graphic_info_text(graphic_timespent_reduction_info)
        update_graphic_figure(figure)
    
    st.plotly_chart(st.session_state.graphic_figure, theme=None)       
    button_info = st.button("Chart Analysis") 
    if(button_info):
        # Display the modified text
        st.markdown(st.session_state.graphic_info_text)

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
    
    target_date = st.text_input("insert the reference month using the format (YYYY-MM)")
    placeholder = st.empty()
    answer = None
    if st.button(label="Generate Optimization"):
        if target_date:
            pattern = r"^\d{4}-\d{2}$"
            if( not (re.match(pattern, target_date))):
                placeholder.warning('Wrong date format', icon="⚠️")
                st.stop()
            input_data = json.dumps({"data": json.loads(data.to_json()), "target_date": target_date})
            api = ProductionConversionCosts()
            answer = api.call(input_data)
    if(answer):
        json_answer = json.loads(answer)
        df_answer = pd.DataFrame.from_dict(ast.literal_eval(json_answer))
        df_answer = df_answer[df_answer["RunTimeHrs"] != 0]
        df_answer.reset_index(inplace = True)
 
        st.dataframe(df_answer)
        target_date = uploaded_file.name.split("_")[0]
        month=int(target_date[5:7])
        setup_time = 0.08
        figure = plot_allocation_plotly(df_pred=df_answer, month=month, setup_time=setup_time)
        st.plotly_chart(figure, theme= None)
        
        st.header("Production Costs", divider=True)
        api = ProductionCostEstimator()
        
        answer_estimator = api.call(json.dumps(json.loads(df_answer.to_json())) )
        answer_estimator = json.loads(json.loads(answer_estimator))['result']
        df_result = pd.DataFrame(answer_estimator)
        
        df_result['Material'] = df_result['Material'].astype(str).str.split('.').str[0]
        df_result['Material'] = df_result['Material'].astype('str')

        df_result['CostCalculated'] = df_result['planQuantityWeight'] * df_result['predictions']
        df_result['CostCalculated'] = np.round(df_result['CostCalculated'],2)
        total_cost = np.round(df_result['CostCalculated'].sum(), 2)
        col1, col2, col3 = st.columns(3)
        with col1:
            pass
        with col2:
            st.metric(label="Total Cost of Production", value=format_large_number(total_cost) + ' Lbs')
        with col3:
            pass
        figure = plot_cost_by_line_material(df_result)
        st.plotly_chart(figure, theme= None)

        col1, col2 = st.columns(2)
        with col1:
            figure = plot_cost_by_material(df_result) 
            st.plotly_chart(figure, theme= None)
        with col2:
            figure = plot_cost_by_machine(df_result)
            st.plotly_chart(figure, theme= None)
        
