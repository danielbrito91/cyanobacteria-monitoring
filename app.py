import streamlit as st
import pandas as pd
import yaml
import json

from src.report.plot_predicted_values import plot_predicted_values

st.title("Cyanobacteria Monitoring - Guaíba Lake")


with open("params.yaml") as config_file:
    config = yaml.safe_load(config_file)

sheet_pred = "https://docs.google.com/spreadsheets/d/1HL9PO6TMQRHW3Z641zERfDRrscGpgXUf6ErpMOEUVLc/edit#gid=0"
url_pred = sheet_pred.replace('/edit#gid=', '/export?format=csv&gid=')

predicted_values = pred = pd.read_csv(url_pred, encoding="latin1", decimal=",")
last_value = predicted_values.tail(1)

st.header("Last predicted value")
st.markdown(
    f"""
    **{round(last_value['y_pred'].values[0])} cells mL-1 on {last_value['date'].values[0]} ({pd.to_datetime(last_value['date']).dt.day_name().values[0]}).**
    """
)

fig = plot_predicted_values("params.yaml")
st.plotly_chart(fig, use_container_width=True)


@st.cache
def convert_df(df):
    return df.to_csv().encode("utf-8")


#csv = convert_df(predicted_values)

#st.download_button(
#    label="Download predictions as CSV",
#    data=csv,
#    file_name="predictions.csv",
#    mime="text/csv",
#)

st.header("About the project")

mae_file = open(config["evaluate"]["metrics_file"])
mae = json.load(mae_file)

st.markdown(
    f"""
A machine learning model to predict the cyanobacteria concentration at Guaíba Lake
using Sentinel 2A data.

Mean absolute error of {round(mae["mae"])} cells/mL
"""
)
