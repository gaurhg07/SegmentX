import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# PAGE TITLE
st.title("ClusterVibe 🚀")
st.subheader("AI Customer Segmentation Platform")

# FILE UPLOAD
uploaded_file = st.file_uploader(
    "Upload Customer CSV File",
    type=["csv"]
)

# IF FILE EXISTS
if uploaded_file is not None:

    # READ CSV
    data = pd.read_csv(uploaded_file)

    # SHOW DATA
    st.write("Customer Dataset")
    st.dataframe(data)

    # SELECT FEATURES
    x = data.iloc[:, [1, 2]]

    # NUMBER OF CLUSTERS
    clusters = st.slider(
        "Select Number of Clusters",
        2,
        10,
        3
    )

    # AI MODEL
    model = KMeans(n_clusters=clusters)

    # PREDICT CLUSTERS
    data["Cluster"] = model.fit_predict(x)

    # SHOW RESULT
    st.write("Clustered Customers")
    st.dataframe(data)

    # GRAPH
    fig, ax = plt.subplots()

    scatter = ax.scatter(
        data.iloc[:, 1],
        data.iloc[:, 2],
        c=data["Cluster"]
    )

    plt.xlabel("Income")
    plt.ylabel("Spending")

    st.pyplot(fig)

    # INSIGHTS
    st.success("AI Segmentation Complete ✅")
