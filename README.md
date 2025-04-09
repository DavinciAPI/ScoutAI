# ScoutAI
🏆 Multi-Sport Talent Scout Pro - GitHub Documentation
📌 Project Overview
Multi-Sport Talent Scout Pro is a Streamlit web app that evaluates and compares athletes across various sports using uploaded or sample CSV data. It leverages AI-powered tools like regression and clustering to identify top talent, profile players, and predict future performance.
✨ Features
- Upload or use default datasets for basketball, football, tennis, and Formula 1.
- Calculates a weighted, normalized talent score for each player.
- Displays top-ranked players with customizable ranking.
- Compare players side-by-side with radar charts and metric tables.
- Predict talent score using a real-time regression model trained on the current dataset.
- Cluster players into talent archetypes using KMeans.
- Explore individual player profiles and (optional) historical performance trends.
📦 Requirements
Install the required packages using:

```
pip install streamlit pandas scikit-learn plotly
```
🚀 How to Run
Run the Streamlit app from your terminal:

```
streamlit run app.py
```

🧠 How AI Talent Prediction Works
The app uses a Linear Regression model trained on the current dataset to predict a player's talent score. It uses raw metrics as input features and the calculated talent score as the prediction target.

Each time a user uploads or loads a dataset, the model is re-trained in real-time.
🔍 How Clustering Works
KMeans clustering is used to group players into 3 default 'talent archetypes' based on their normalized performance scores. These clusters help users understand relative performance tiers.
🙌 Credits
Created by MR.ROBOT Team to support scouts, coaches, and data analysts.
