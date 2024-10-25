import streamlit as st

# Page configuration
st.set_page_config(page_title="AI MVPs", layout="wide")

# Main title
st.title("AI MVPs - Data Science Projects")

# Introductory description
st.write("""
Welcome to the Celanese AI MVP platform! Here, you will find an overview of the data science and AI projects 
currently in development, aimed at improving safety, operational efficiency, and reducing costs across our plants. 
These projects focus on delivering actionable insights to optimize industrial processes.
""")

# Project 1: Root Cause Analysis on Incidents
st.subheader("🔍 Root Cause Analysis on Incidents")
st.write("""
This project aims to enhance safety in Celanese plants by identifying the root causes of incidents and providing insights 
to support preventive measures. Advanced data analytics are used to highlight similarities between incidents across 
different plants, helping to reduce future risks.
""")
st.write("- **Objective:** Reduce future incidents through insights on root causes and trends.")
st.write("- **Expected outcomes:** Identification of root causes, preventive metrics, and improvements in operational safety.")

# Project 2: Incident Probability Score
st.subheader("📊 Incident Probability Score")
st.write("""
Focusing on safety and predictive maintenance, this project aims to predict equipment failures by monitoring anomalous 
patterns in operations. By generating metrics and detecting anomalies, it helps to avoid production losses and high maintenance costs.
""")
st.write("- **Objective:** Predict failures and detect operational anomalies.")
st.write("- **Expected outcomes:** Increased safety, reduced failures, and lower maintenance costs.")

# Project 3: Control Tower - Boilers
# st.subheader("🔥 Control Tower - Boilers")
# st.write("""
# This project leverages process data from the Clear Lake plant to monitor the efficiency of the main boilers in real-time. 
# Predictive models allow the detection of abnormal conditions early, enabling quick corrective actions.
# """)
# st.write("- **Objective:** Monitor and predict boiler efficiency.")
# st.write("- **Expected outcomes:** Early detection of anomalies and optimized preventive maintenance.")

# Project 4: Production Cost Reduction
st.subheader("💡 Production Cost Reduction")
st.write("""
By analyzing production conversion data, this project aims to optimize plant capacity and reduce operational costs. 
It provides actionable recommendations to improve efficiency and reduce resource consumption.
""")
st.write("- **Objective:** Identify cost reduction opportunities based on plant capacity and product lines.")
st.write("- **Expected outcomes:** Cost optimization and increased production efficiency.")

# Project 5: Predict Conversion Loss
st.subheader("📉 Predict Conversion Loss")
st.write("""
This project uses advanced AI analytics to predict potential conversion losses, enabling proactive actions to minimize waste 
and improve production efficiency across multiple plants.
""")
st.write("- **Objective:** Predict and reduce conversion losses.")
st.write("- **Expected outcomes:** Improved operational efficiency and enhanced plant performance.")

# Footer
st.write("---")
st.write("Explore each project from the menu on the left for more details.")
