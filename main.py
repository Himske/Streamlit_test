import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()
interactive = st.container()


st.markdown(
    """
    <style>
    .main {
        background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache
def get_data(filename, nrows):
    return pd.read_csv(filename, nrows=nrows)


with header:
    st.title("Welcome to my awesome data science project!")
    st.text("In this project I look into the transactions of taxis in NYC.")

with dataset:
    st.header("NYC taxi dataset")
    st.text("I found this dataset on https://data.cityofnewyork.us/Transportation/2020-Yellow-Taxi-Trip-Data/kxp8-n2sj")

    taxi_data = get_data("data/2020_Yellow_Taxi_Trip_Data.csv", 1000)

    st.subheader("Pick-up location ID distribution on the NYC dataset")
    pulocation_dist = pd.DataFrame(taxi_data["PULocationID"].value_counts()).head(50)
    st.bar_chart(pulocation_dist)

with features:
    st.header("The features I created")

    st.markdown("* **first feature:** I created this feature because of this... I calculated it using this logic...")
    st.markdown("* **second feature:** I created this feature because of this... I calculated it using this logic...")

with model_training:
    st.header("Time to train the model")
    st.text("Here you get to choose the hyperparameters of the model and see how the performance changes")

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider("What should be the max_depth of the model?", min_value=10, max_value=100, value=20,
                               step=10)

    n_estimators = sel_col.selectbox("How many trees should there be?", options=[100, 200, 300, "No limit"], index=0)

    input_feature = sel_col.selectbox("Which feature should be used as the input feature?",
                                      options=taxi_data.columns, index=0)

    if n_estimators == "No limit":
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    X = taxi_data[[input_feature]]
    y = taxi_data[["trip_distance"]]

    regr.fit(X, y)
    prediction = regr.predict(y)

    disp_col.subheader("Mean absolute error of the model is:")
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader("Mean squared error of the model is:")
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader("R squared error of the model is:")
    disp_col.write(r2_score(y, prediction))

with interactive:
    st.title("A closer look at the data")

    fig = go.Figure(data=go.Table(
        header=dict(values=list(taxi_data[["tpep_pickup_datetime", "trip_distance", "total_amount"]].columns),
                    fill_color="#FD8E72", align="left"),
        cells=dict(values=[taxi_data.tpep_pickup_datetime, taxi_data.trip_distance, taxi_data.total_amount],
                   fill_color="#E5ECF6", align="left"),
        columnwidth=[2, 1, 1]))

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    st.write(fig)
