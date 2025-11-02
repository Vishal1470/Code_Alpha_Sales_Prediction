import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# App title
st.set_page_config(page_title="Sales Prediction App", page_icon="💰", layout="centered")
st.title("📈 Sales Prediction using Advertising Data")

# Load trained model
try:
    with open('model/sales_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ Model file not found! Please run 'train_model.py' first.")
    st.stop()

# Sidebar inputs
st.sidebar.header("Enter Advertising Spend")
tv = st.sidebar.number_input("💻 TV Advertising Spend", 0.0)
radio = st.sidebar.number_input("📻 Radio Advertising Spend", 0.0)
newspaper = st.sidebar.number_input("📰 Newspaper Advertising Spend", 0.0)

# Predict sales for entered values
if st.sidebar.button("🔮 Predict Sales"):
    features = np.array([[tv, radio, newspaper]])
    prediction = model.predict(features)[0]
    st.success(f"📊 Predicted Sales: **{prediction:.2f} units**")

# Generate random products (for demo)
products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
spend_data = pd.DataFrame({
    'TV': np.random.uniform(50, 300, 5),
    'Radio': np.random.uniform(20, 100, 5),
    'Newspaper': np.random.uniform(10, 80, 5)
})
spend_data['Predicted Sales'] = model.predict(spend_data[['TV', 'Radio', 'Newspaper']])
spend_data['Product'] = products

# Show product sales table
st.subheader("📦 Predicted Sales for Different Products")
st.dataframe(spend_data)

# Show bar graph
st.subheader("📊 Product-wise Sales Prediction")
fig, ax = plt.subplots()
ax.bar(spend_data['Product'], spend_data['Predicted Sales'], color='skyblue')
ax.set_xlabel('Product')
ax.set_ylabel('Predicted Sales')
ax.set_title('Predicted Sales by Product')
st.pyplot(fig)

# Best performing product
best_product = spend_data.loc[spend_data['Predicted Sales'].idxmax(), 'Product']
best_sales = spend_data['Predicted Sales'].max()
st.success(f"🏆 Highest predicted sales: **{best_product}** with **{best_sales:.2f} units**")


