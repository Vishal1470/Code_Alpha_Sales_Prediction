💰 Sales Prediction using Python
🧭 Overview

This project predicts future product sales based on advertising spend across different platforms such as TV, Radio, and Newspaper.
It uses Machine Learning (Linear Regression) to forecast sales and provides insights through interactive graphs using Streamlit.

📁 Folder Structure
Code_Alpha_Sales_Prediction/

│

├── dataset/

│   └── advertising.csv               # Dataset file

│

├── model/

│   └── sales_model.pkl               # Trained ML model

│

├── app.py                            # Streamlit web app

├── train_model.py                    # Model training script

├── requirements.txt                  # Required libraries

└── README.md                         # Project documentation

⚙️ Setup Instructions
1️⃣ Clone or Create Project Folder

Create a new folder named Code_Alpha_Sales_Prediction and open it in VS Code or any IDE.

2️⃣ Install Dependencies

Run this command in your terminal:

pip install -r requirements.txt


If you don’t have a requirements.txt, create one with:

streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn

3️⃣ Add Dataset

Download the dataset (advertising.csv) and place it inside the dataset/ folder.
The dataset should have the following columns:

TV, Radio, Newspaper, Sales

🧠 Model Training

Run this command to train your model:

python train_model.py


This script will:

Load and clean the dataset

Train a Linear Regression model

Display graphs for correlation and prediction accuracy

Save the trained model as model/sales_model.pkl

💻 Running the Web App

After training, run the Streamlit app:

streamlit run app.py

The app will:

✅ Take advertising spend inputs (TV, Radio, Newspaper)

✅ Predict sales using the trained model

✅ Display predicted sales for multiple products

✅ Show product-wise bar chart visualization

✅ Highlight the best-performing product

📊 Insights Delivered

Understand how advertising channels influence sales

Visualize actual vs. predicted sales patterns

Identify the most effective marketing platform

Suggest which product may perform best based on advertising budget

🧩 Example Output

Dashboard Includes:

Input sliders for ad spend

Predicted sales result

Interactive table and bar graph

Highlight of highest predicted sales

📜 Technologies Used

Tool	Purpose

Python	Programming Language

Pandas / NumPy	Data Processing

Scikit-learn	Machine Learning

Matplotlib / Seaborn	Data Visualization

Streamlit	Web App Framework

🏁 Future Enhancements

Include product images in dashboard

Use time-series forecasting models (ARIMA / Prophet)

Add real-time data updates from CSV or API

👨‍💻 Author

Vishal Baburao Patil

G. H. Raisoni College of Engineering and Management, Jalgaon
