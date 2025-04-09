import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np

# ------------------- Configuration -------------------
SPORT_CONFIG = {
    "basketball": {
        "metrics": ["vertical_jump_cm", "three_point_percent", "assists_per_game"],
        "weights": [0.4, 0.3, 0.3],
        "file": "basketball_data.csv",
        "display_names": {
            "vertical_jump_cm": "Vertical Jump (cm)",
            "three_point_percent": "3-Point %",
            "assists_per_game": "Assists/Gm"
        },
        "id_column": "player_name"
    },
    "football": {
        "metrics": ["sprint_40m_sec", "pass_accuracy_percent", "goals_last_season"],
        "weights": [0.3, 0.4, 0.3],
        "file": "football_data.csv",
        "display_names": {
            "sprint_40m_sec": "40m Sprint (sec)",
            "pass_accuracy_percent": "Pass Accuracy %",
            "goals_last_season": "Goals/Season"
        },
        "id_column": "player_name"
    },
    "tennis": {
        "metrics": ["serve_speed_kmh", "first_serve_percent", "break_points_saved_percent"],
        "weights": [0.4, 0.4, 0.2],
        "file": "tennis_data.csv",
        "display_names": {
            "serve_speed_kmh": "Serve Speed (km/h)",
            "first_serve_percent": "1st Serve %",
            "break_points_saved_percent": "Break Points Saved %"
        },
        "id_column": "player_name"
    },
    "formula1": {
        "metrics": ["qualifying_lap_time_sec", "race_lap_consistency_stddev", "overtakes_per_race"],
        "weights": [0.5, 0.3, 0.2],
        "file": "f1_data.csv",
        "display_names": {
            "qualifying_lap_time_sec": "Quali Lap Time (sec)",
            "race_lap_consistency_stddev": "Lap Consistency (stddev)",
            "overtakes_per_race": "Overtakes/Race"
        },
        "id_column": "driver_name"
    }
}

# ------------------- Utility Functions -------------------
def normalize(series, is_higher_better=True):
    """Normalize a pandas Series to a 0-100 scale."""
    if is_higher_better:
        return (series - series.min()) / (series.max() - series.min()) * 100
    else:
        return (series.max() - series) / (series.max() - series.min()) * 100

def load_data(sport):
    """Load data from uploaded file or use default CSV."""
    uploaded_file = st.file_uploader(f"Upload your {sport} data (CSV)", type="csv", key=f"{sport}_upload")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return pd.DataFrame()
    else:
        default_path = Path(__file__).parent / SPORT_CONFIG[sport]["file"]
        try:
            df = pd.read_csv(default_path)
            st.info(f"ℹ Using default {sport} dataset.")
        except Exception as e:
            st.error(f"Could not load default file: {default_path}\nError: {e}")
            return pd.DataFrame()
    return df

def calculate_scores(df, sport):
    """Calculate talent score based on weighted, normalized metrics."""
    config = SPORT_CONFIG[sport]
    df["talent_score"] = 0
    for metric, weight in zip(config["metrics"], config["weights"]):
        # Consider higher values as better unless the metric name suggests time or variability.
        is_higher_better = not ("time" in metric or "stddev" in metric)
        df[f"{metric}_score"] = normalize(df[metric], is_higher_better)
        df["talent_score"] += df[f"{metric}_score"] * weight
    df = df.sort_values("talent_score", ascending=False).reset_index(drop=True)
    return df

def assign_clusters(df, sport, n_clusters=3):
    """Apply KMeans clustering on the normalized metric scores to classify talent archetypes."""
    config = SPORT_CONFIG[sport]
    score_cols = [f"{m}_score" for m in config["metrics"]]
    if df.empty or len(df) < n_clusters:
        df["cluster"] = np.nan
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df["cluster"] = kmeans.fit_predict(df[score_cols])
    return df

def train_regression_model(df, sport):
    """Train a basic linear regression to predict talent score from raw metrics."""
    config = SPORT_CONFIG[sport]
    X = df[config["metrics"]]
    y = df["talent_score"]
    model = LinearRegression()
    model.fit(X, y)
    return model

def plot_radar(df, player1, player2, sport):
    """Generate a radar chart comparing two players based on their normalized scores."""
    config = SPORT_CONFIG[sport]
    metrics = config["metrics"]
    display_names = [config["display_names"][m] for m in metrics]
    
    p1 = df[df[config["id_column"]] == player1].iloc[0]
    p2 = df[df[config["id_column"]] == player2].iloc[0]
    
    fig = go.Figure()
    for player, data in zip([player1, player2], [p1, p2]):
        fig.add_trace(go.Scatterpolar(
            r=[data[f"{m}_score"] for m in metrics],
            theta=display_names,
            fill='toself',
            name=player
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="Performance Radar")
    st.plotly_chart(fig, use_container_width=True)

def compare_players(df, p1, p2, sport):
    """Display a side-by-side comparison of two players."""
    config = SPORT_CONFIG[sport]
    id_col = config["id_column"]
    metrics = config["metrics"]
    
    p1_data = df[df[id_col] == p1].iloc[0]
    p2_data = df[df[id_col] == p2].iloc[0]
    
    comp_df = pd.DataFrame({
        "Metric": [config["display_names"][m] for m in metrics],
        p1: [p1_data[m] for m in metrics],
        p2: [p2_data[m] for m in metrics]
    })

    st.subheader("📊 Metric Comparison")
    st.dataframe(comp_df.set_index("Metric"), use_container_width=True)

    st.subheader("📈 Radar Chart")
    plot_radar(df, p1, p2, sport)

    st.subheader("🌟 Talent Scores")
    col1, col2 = st.columns(2)
    with col1: st.metric(p1, round(p1_data["talent_score"], 1))
    with col2: st.metric(p2, round(p2_data["talent_score"], 1))

def show_player_profile(df, sport, player):
    """Display the player profile along with historical trend charts if available."""
    config = SPORT_CONFIG[sport]
    id_col = config["id_column"]
    player_data = df[df[id_col] == player]
    
    st.subheader(f"Profile: {player}")
    st.write("**Basic Details:**")
    st.write(player_data.iloc[0][[id_col, "talent_score"]])
    
    # Display cluster label (if available)
    if "cluster" in player_data.columns:
        st.write(f"**Talent Cluster:** {player_data.iloc[0]['cluster']}")
    
    # Check if historical data is available (assumes a column 'season' or 'year')
    if "season" in player_data.columns or "year" in player_data.columns:
        time_col = "season" if "season" in player_data.columns else "year"
        st.write("### Historical Trends")
        metrics = config["metrics"]
        # Create a line chart for each metric over the seasons/years
        for metric in metrics:
            trend_df = player_data.sort_values(time_col)[[time_col, metric]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend_df[time_col], y=trend_df[metric],
                                     mode='lines+markers', name=config["display_names"][metric]))
            fig.update_layout(title=f"{config['display_names'][metric]} Over Time", xaxis_title=time_col.capitalize(), yaxis_title=metric)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical trend data available for this player.")

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Multi-Sport Talent Scout Pro", layout="wide")
st.title("🏆 Multi-Sport Talent Scout Pro")

# ---- Select Sport & Load Data ----
sport = st.selectbox("Select a Sport", list(SPORT_CONFIG.keys()))
df = load_data(sport)

if not df.empty:
    # Calculate talent scores
    df = calculate_scores(df, sport)

    # AI Clustering: assign players to clusters (talent archetypes)
    df = assign_clusters(df, sport, n_clusters=3)

    # Display top talents with a custom Rank column
    top_df = df.head(10).copy()
    top_df.insert(0, "Rank", [f"Rank {i+1}" for i in range(len(top_df))])
    st.subheader(f"⭐ Top {sport.title()} Talents")
    st.dataframe(top_df.style.highlight_max(axis=0, subset=top_df.columns[2:]), use_container_width=True)

    st.divider()
    # ---- Compare Two Players ----
    st.subheader("🔍 Compare Players")
    id_col = SPORT_CONFIG[sport]["id_column"]
    players = df[id_col].unique()
    col1, col2 = st.columns(2)
    with col1:
        p1 = st.selectbox("Select Player 1", players, key="compare_p1")
    with col2:
        p2 = st.selectbox("Select Player 2", [p for p in players if p != p1], key="compare_p2")
    if p1 and p2:
        compare_players(df, p1, p2, sport)

    st.divider()
    # ---- AI Talent Prediction ----
    st.subheader("🤖 AI Talent Prediction")
    st.write("Enter raw metric values to predict the talent score.")
    config = SPORT_CONFIG[sport]
    input_data = {}
    for metric in config["metrics"]:
        input_data[metric] = st.number_input(f"{config['display_names'][metric]}", value=float(df[metric].mean()))
    if st.button("Predict Talent Score"):
        model = train_regression_model(df, sport)
        # Prepare input as a DataFrame with one record
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Talent Score: {round(prediction, 1)}")

    st.divider()
    # ---- Player Profile and Historical Trend ----
    st.subheader("👤 Player Profile")
    profile_player = st.selectbox("Select a Player to view Profile", players, key="profile_player")
    if profile_player:
        show_player_profile(df, sport, profile_player)

# ---- Sidebar Sample Data Download ----
st.sidebar.markdown("### ⬇ Download Sample Data")
sample_sport = st.sidebar.selectbox("Choose Sport", list(SPORT_CONFIG.keys()), key="sample_sport")
if st.sidebar.button("Download Sample"):
    path = Path(__file__).parent / SPORT_CONFIG[sample_sport]["file"]
    try:
        sample = pd.read_csv(path)
        st.sidebar.download_button(
            label="📁 Download Sample CSV",
            data=sample.to_csv(index=False).encode('utf-8'),
            file_name=f"sample_{sample_sport}.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.sidebar.error(f"Error loading sample file: {e}")
