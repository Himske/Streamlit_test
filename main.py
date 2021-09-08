import pandas as pd
import streamlit as st


header = st.container()

dataset = st.container()

features = st.container()

model_training = st.container()

with header:
    st.title("Welcome to my awesome data science project!")
    st.text("In this project I look into the transactions of taxis in NYC.")

with dataset:
    st.header("NYC taxi dataset")
    st.text("I found this dataset on https://data.cityofnewyork.us/Transportation/2020-Yellow-Taxi-Trip-Data/kxp8-n2sj")

    taxi_data = pd.read_csv("data/2020_Yellow_Taxi_Trip_Data.csv")
    st.write(taxi_data.head())

    pulocation_dist = pd.DataFrame(taxi_data["PULocationID"].value_counts())
    st.bar_chart(pulocation_dist)

with features:
    st.header("The features I created")

with model_training:
    st.header("Time to train the model")
    st.text("Here you get to choose the hyperparameters of the model and see how the performance changes")
