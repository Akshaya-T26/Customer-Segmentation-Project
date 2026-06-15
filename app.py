import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("Customer Segmentation Dashboard")

df = pd.read_csv("customers.csv")

st.subheader("Dataset Preview")
st.dataframe(df)

X = df[["Annual_Income", "Spending_Score"]]

k = st.slider("Select Number of Clusters", 2, 6, 3)

model = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X)

st.subheader("Cluster Analysis")

cluster_summary = df.groupby("Cluster")[["Annual_Income", "Spending_Score"]].mean()
st.dataframe(cluster_summary)

def label_cluster(row):
    if row["Annual_Income"] > 60000 and row["Spending_Score"] > 60:
        return "High Value Customer"
    elif row["Annual_Income"] < 30000 and row["Spending_Score"] < 40:
        return "Low Value Customer"
    else:
        return "Average Customer"

df["Segment"] = df.apply(label_cluster, axis=1)

st.subheader("Segmented Data")
st.dataframe(df)

st.subheader("Customer Segments Visualization")

fig1 = px.scatter(
    df,
    x="Annual_Income",
    y="Spending_Score",
    color="Cluster",
    hover_data=["CustomerID", "Age", "Segment"],
    title="Customer Segments"
)

st.plotly_chart(fig1, use_container_width=True)

st.subheader("Segment Distribution")

fig2 = px.pie(
    df,
    names="Segment",
    title="Customer Segment Distribution"
)

st.plotly_chart(fig2, use_container_width=True)