import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# LOAD MODELS
# ==============================

kmeans = joblib.load("../models/kmeans.pkl")
scaler = joblib.load("../models/scaler.pkl")

# ==============================
# BUILD SIMILARITY MATRIX (DYNAMIC)
# ==============================

@st.cache_data
def build_similarity_matrix():
    df = pd.read_csv("https://drive.google.com/uc?id=1xVV5c_X4ZGZd3QEXQt-u_5JPwzmfizT6")

    # Clean data
    df = df.dropna(subset=["CustomerID", "Description"])
    df = df[df["Quantity"] > 0]

    # Create pivot table
    pivot = df.pivot_table(
        index="Description",
        columns="CustomerID",
        values="Quantity",
        fill_value=0
    )

    similarity = cosine_similarity(pivot)
    product_list = list(pivot.index)

    return similarity, product_list

similarity, product_list = build_similarity_matrix()

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Shopper Spectrum Dashboard",
    page_icon="🛒",
    layout="wide"
)

# ==============================
# CUSTOM CSS
# ==============================

st.markdown("""
<style>
.big-title {
    font-size: 44px;
    font-weight: 800;
    color: #0f172a;
}
.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-top: 10px;
}
.kpi-box {
    padding: 22px;
    border-radius: 16px;
    background: linear-gradient(135deg, #f8fafc, #eef2ff);
    text-align: center;
    font-size: 22px;
    font-weight: 700;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}
.card {
    padding: 22px;
    border-radius: 16px;
    background: #ffffff;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR
# ==============================

st.sidebar.title("🧭 Navigation")
menu = st.sidebar.radio("Go to", ["Dashboard", "Customer Segmentation", "Product Recommendation"])

# ==============================
# HEADER
# ==============================

st.markdown('<div class="big-title">🛒 Shopper Spectrum Dashboard</div>', unsafe_allow_html=True)
st.markdown("### Customer Segmentation & Product Recommendation System")

# ==============================
# DASHBOARD
# ==============================

if menu == "Dashboard":
    st.subheader("📊 Business Intelligence Dashboard")

    # ===== KPI ROW =====
    col1, col2, col3, col4 = st.columns(4)

    col1.markdown('<div class="kpi-box">📦 Products<br>~4000+</div>', unsafe_allow_html=True)
    col2.markdown('<div class="kpi-box">👥 Customers<br>~4300+</div>', unsafe_allow_html=True)
    col3.markdown('<div class="kpi-box">🧠 ML Models<br>2</div>', unsafe_allow_html=True)
    col4.markdown('<div class="kpi-box">⚡ System<br>Real-Time</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ===== TABS =====
    tab1, tab2, tab3 = st.tabs(["📊 Segments Overview", "📈 RFM Insights", "🧠 Business Interpretation"])

    # ===== TAB 1 =====
    with tab1:
        cluster_names = ["High-Value", "Regular", "At-Risk", "Occasional"]
        cluster_counts = [2171, 1326, 828, 13]

        df_clusters = pd.DataFrame({
            "Segment": cluster_names,
            "Customers": cluster_counts
        })

        fig = px.pie(
            df_clusters,
            names="Segment",
            values="Customers",
            title="Customer Segmentation Distribution",
            hole=0.4
        )

        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 High-Value and Regular customers form the majority of the business revenue base.")

    # ===== TAB 2 =====
    with tab2:
        st.subheader("📈 Sample RFM Distributions")

        recency = np.random.gamma(2, 100, 1000)
        frequency = np.random.poisson(5, 1000)
        monetary = np.random.gamma(2, 500, 1000)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.plotly_chart(px.histogram(recency, title="Recency Distribution"), use_container_width=True)
        with col2:
            st.plotly_chart(px.histogram(frequency, title="Frequency Distribution"), use_container_width=True)
        with col3:
            st.plotly_chart(px.histogram(monetary, title="Monetary Distribution"), use_container_width=True)

        st.info("💡 These distributions help understand customer purchasing behavior patterns.")

    # ===== TAB 3 =====
    with tab3:
        st.subheader("🧠 Business Insights from Segmentation")

        st.write("🔹 **High-Value Customers**: Frequent buyers, high spenders — main revenue drivers.")
        st.write("🔹 **Regular Customers**: Stable buyers — can be upsold to high-value.")
        st.write("🔹 **Occasional Customers**: Low engagement — target with offers.")
        st.write("🔹 **At-Risk Customers**: Haven't purchased recently — need retention campaigns.")

        st.markdown("---")

        st.write("📌 **Business Strategy Recommendations:**")
        st.write("• Focus loyalty programs on High-Value customers")
        st.write("• Offer discounts to Regular & Occasional customers")
        st.write("• Re-engage At-Risk customers with email campaigns")

    st.markdown("---")
    st.markdown("📌 This dashboard simulates a real-world business intelligence system built using Machine Learning.")

# ==============================
# CUSTOMER SEGMENTATION
# ==============================

elif menu == "Customer Segmentation":
    st.subheader("👥 Customer Segmentation Predictor")

    col1, col2, col3 = st.columns(3)

    with col1:
        r = st.number_input("Recency (days)", min_value=0, value=100)
    with col2:
        f = st.number_input("Frequency (purchases)", min_value=1, value=5)
    with col3:
        m = st.number_input("Monetary (total spend)", min_value=0.0, value=1000.0)

    if st.button("🔮 Predict Segment"):
        user_data = np.array([[r, f, m]])
        user_scaled = scaler.transform(user_data)
        cluster = kmeans.predict(user_scaled)[0]

        segment_map = {
            0: "High-Value Customer 💎",
            1: "Regular Customer 🙂",
            2: "Occasional Customer 🛒",
            3: "At-Risk Customer ⚠️"
        }

        st.success(f"### 🧠 Prediction Result: **{segment_map.get(cluster)}**")

# ==============================
# PRODUCT RECOMMENDATION
# ==============================

elif menu == "Product Recommendation":
    st.subheader("🛍 Product Recommendation System")

    product = st.selectbox("Select a product", product_list)

    if st.button("✨ Get Recommendations"):
        idx = product_list.index(product)
        scores = list(enumerate(similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

        st.markdown("### ✅ Recommended Products:")

        for i, (index, score) in enumerate(scores, 1):
            st.write(f"**{i}.** {product_list[index]}")

# ==============================
# FOOTER
# ==============================

st.markdown("---")
st.markdown("🚀 Developed as part of **Shopper Spectrum ML Project** | End-to-End Machine Learning System")
