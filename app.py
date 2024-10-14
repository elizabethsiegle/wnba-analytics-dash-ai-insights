import base64
from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path

import plotly.graph_objects as go
# import plotly.express as px
import requests
import streamlit as st
from streamlit_plotly_events import plotly_events



load_dotenv()

# Cloudflare Workers AI setup
ACCOUNT_ID = os.getenv('CF_ACCOUNT_ID')
AUTH_TOKEN = os.getenv('CF_AUTH_TOKEN')

# Updated Page configuration
st.set_page_config(page_title="WNBA Player Analytics Dashboard && AI Insights", page_icon="🏀", layout="wide")

# Enhanced Custom CSS with gradient background, hover effects, and sticky footer
st.markdown("""
<style>
    body {
        background: linear-gradient(120deg, #2c3e50 0%, #3498db 100%);
        color: white;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    .stPlotlyChart {
        transition: transform 0.3s ease-in-out;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-radius: 10px;
        overflow: hidden;
    }
    .stPlotlyChart:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
    }
    .sticky-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(44, 62, 80, 0.8);
        color: white;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
    }
    .main-content {
        margin-bottom: 50px;
    }
    h1, h2, h3 {
        color: #ecf0f1;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
    }
    .chart-section, .stats-section, .player-info-section, .player-stats-section {
        background-color: rgba(52, 73, 94, 0.7);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .chart-section {
        background-color: rgba(41, 128, 185, 0.7);
    }
    .stats-section {
        background-color: rgba(192, 57, 43, 0.7);
    }
    .player-info-section {
        background-color: rgba(39, 174, 96, 0.7);
    }
    .player-stats-section {
        background-color: rgba(142, 68, 173, 0.7);
    }
</style>
""", unsafe_allow_html=True)

# Image display function
def encode_image(img_path):
    return base64.b64encode(Path(img_path).read_bytes()).decode()

def generate_insights(data):
    # Select top 10 players by points
    top_players = data.nlargest(10, 'PTS')
    
    # Select key statistics
    key_stats = ['Player', 'Team', 'Pos', 'PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%', '3P%']
    summary_data = top_players[key_stats]
    
    prompt = f"""
    Analyze the following WNBA player statistics:
    {summary_data.to_string(index=False)}

    Provide 3 insights, opinions, observations or hot takes about this data, focusing on scoring, efficiency, and overall performance trends with no preamble.
    For each insight:
    1. State the observation clearly.
    2. Provide specific data points from the table to support your observation.
    3. Explain the significance of this observation in the context of basketball.

    Format your response as a numbered list, with each insight clearly separated.
    Do not make any claims that cannot be directly supported by the data provided.
    If you're unsure about a claim, state that it's a possibility rather than a certainty.
    """
    
    response = requests.post(
        f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/meta/llama-3.1-8b-instruct-fast",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        json={
            "messages": [
                {"role": "system", "content": "You are a women's basketball analyst known for providing helpful and insightful opinions on WNBA player statistics. You only make claims that can be directly supported by the data provided."},
                {"role": "user", "content": prompt}
            ],
        }
    )
    result = response.json()
    return result['result']['response']

# Display WNBA logo and dashboard information
st.markdown(
    f"""
    <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">
        <img src='data:image/png;base64,{encode_image('wnba-logo.png')}' class='img-fluid' style="max-width: 200px; margin-bottom: 20px;">
        <h1 style="margin-bottom: 20px; font-size: 3em;">WNBA Player Analytics Dashboard && AI Insights</h1>
        <p style="max-width: 800px; margin-bottom: 20px; font-size: 1.8em; line-height: 1.6;">
            Explore comprehensive WNBA player statistics with this interactive dashboard.
            <br><strong>Data Source:</strong> <a href="https://www.basketball-reference.com/" target="_blank">Basketball-reference.com</a>
        </p>
        <p style="font-style: italic; color: #f1c40f; font-size: 1.8em;">
            🔍 Use the sidebar to refine your search
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar controls
st.sidebar.header('Filter Options')
st.html(
    """
<style>
[data-testid="stSidebarContent"] {
    color: white;
    background-color: #B896D4;
}
</style>
"""
)
selected_season = st.sidebar.selectbox('Season', list(reversed(range(1997, 2025))))

# Data fetching function
@st.cache_data
def fetch_player_data(season):
    url = f"https://www.basketball-reference.com/wnba/years/{season}_per_game.html"
    dataframes = pd.read_html(url, header=0)
    df = dataframes[0]
    df = df[df.G != 'G'].fillna(0)  # Remove header rows and fill NaNs
    return df.drop(['G'], axis=1)

player_data = fetch_player_data(selected_season)

def colorize_multiselect_options(teams: list[str], team_colors: dict[str, str]) -> None:
    rules = ""

    for i, team in enumerate(teams):
        color = team_colors.get(team, "#FFFFFF")  # Default to white if team not found in dictionary
        rules += f"""
        .stMultiSelect div[data-baseweb="select"] span[data-baseweb="tag"]:nth-child({i + 1}) {{
            background-color: {color};
            color: white;  /* Ensure the text is readable */
        }}
        """

    # Apply the CSS rules to the page
    st.markdown(f"<style>{rules}</style>", unsafe_allow_html=True)

# Create the teams list, excluding "TOT"
teams = sorted([team for team in player_data.Team.unique() if team != 'TOT'])

# Define specific team colors
team_colors = {
    'NYL': '#6ECEB2',    # Green for NY Liberty
    'ATL': '#C8102E',
    'CON': '#FC4C02',
    'DAL': '#C4D600',
    'IND': '#041E42',
    'LVA': '#010101',
    'MIN': '#236192',
    'SEA': '#2C5234',  # Blue for Seattle Storm
    'PHO': '#CB6015',  # Orange for Phoenix Mercury
    'LAS': '#702F8A',  # Purple for LA Sparks
    'CHI': '#418FDE',  # Light Blue for Chicago Sky
    'WAS': '#C8102E'
    # Add more teams and their respective colors
}
selected_teams = st.sidebar.multiselect('Team', teams, teams)
# Apply the colors to the teams in the multiselect
colorize_multiselect_options(teams, team_colors)

positions = ['C', 'F', 'G', 'F-G', 'C-F']
selected_positions = st.sidebar.multiselect('Position', positions, positions)

# Data filtering
filtered_data = player_data[
    (player_data.Team.isin(selected_teams)) & 
    (player_data.Pos.isin(selected_positions))
].sort_values(by='PTS', ascending=False)  # Sort by points

# Move the pie chart above the displayed data
st.subheader("Top 5 Scorers (Points per Game)")

# Initialize session state for selected player
if 'selected_player' not in st.session_state:
    st.session_state.selected_player = None

# Function to update selected player
def update_selected_player(player):
    st.session_state.selected_player = player

# Create a 2x2 grid layout
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    st.markdown('<div class="chart-section">', unsafe_allow_html=True)
    st.subheader("Top 5 Scorers (Points per Game)")
    # Chart type selector
    chart_type = st.radio("Select chart type:", ("Pie Chart", "Bar Chart"))

    filtered_data['PTS'] = pd.to_numeric(filtered_data['PTS'], errors='coerce')
    top_scorers = filtered_data.nlargest(5, "PTS")
    
    hover_text = [f"{player}<br>Team: {team}<br>Position: {pos}" 
                  for player, team, pos in zip(top_scorers['Player'], top_scorers['Team'], top_scorers['Pos'])]

    # Get team colors for the top scorers
    colors = [team_colors.get(team, '#000000') for team in top_scorers['Team']]

    if chart_type == "Pie Chart":
        fig = go.Figure(data=[go.Pie(
            labels=top_scorers['Player'],
            values=top_scorers['PTS'],
            hovertemplate="<b>%{label}</b><br>" +
                          "Points per Game: %{value:.1f}<br>" +
                          "%{text}" +
                          "<extra></extra>",
            text=hover_text,
            textinfo='label+percent',
            insidetextorientation='radial',
            marker=dict(colors=colors)  # Use team colors for pie slices
        )])
    else:  # Bar Chart
        fig = go.Figure(data=[go.Bar(
            x=top_scorers['Player'],
            y=top_scorers['PTS'],
            hovertemplate="<b>%{x}</b><br>" +
                          "Points per Game: %{y:.1f}<br>" +
                          "%{text}" +
                          "<extra></extra>",
            text=hover_text,
            marker_color=colors  # Use team colors for bars
        )])
        fig.update_xaxes(title_text="Player")
        fig.update_yaxes(title_text="Points per Game")

    fig.update_layout(
        title="Top Scorers",
        height=400,  # Increase the height of the chart
        plot_bgcolor='rgba(0,0,0,0.1)',
        paper_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color="white", size=14),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="white",
            borderwidth=1
        )
    )
    
    selected_points = plotly_events(fig, click_event=True, override_height=400)
    
    if selected_points:
        selected_player = selected_points[0]['label'] if chart_type == "Pie Chart" else selected_points[0]['x']
        update_selected_player(selected_player)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="stats-section">', unsafe_allow_html=True)
    st.markdown("### Chart Filters")
    min_points = st.slider("Minimum Points", min_value=0, max_value=int(top_scorers['PTS'].max()), value=0)
    filtered_top_scorers = top_scorers[top_scorers['PTS'] >= min_points]

    st.markdown("### Quick Stats")
    if not filtered_top_scorers.empty:
        st.metric("Average Points", f"{filtered_top_scorers['PTS'].mean():.1f}")
        highest_scorer = filtered_top_scorers.loc[filtered_top_scorers['PTS'].idxmax()]
        st.metric("Highest Scorer", f"{highest_scorer['Player']} ({highest_scorer['PTS']:.1f} pts)")
        st.metric("Players Shown", f"{len(filtered_top_scorers)}")
    else:
        st.warning("No players match the current filter.")
    
    st.markdown("### Top Scorers")
    # Ensure PTS is numeric and round to 1 decimal place
    filtered_top_scorers['PTS'] = pd.to_numeric(filtered_top_scorers['PTS'], errors='coerce').round(1)

    # Create a custom style function
    def color_team(val):
        color = team_colors.get(val, 'black')  # Default to black if team not found
        return f'color: {color}; font-weight: bold;'

    # Apply the style to the dataframe
    styled_df = filtered_top_scorers[["Player", "PTS", "Team"]].style\
        .set_properties(**{'background-color': '#f0f0f0'})\
        .applymap(color_team, subset=['Player', 'Team'])\
        .applymap(lambda x: 'color: black; font-weight: bold;', subset=['PTS'])\
        .format({'PTS': '{:.1f}'})

    # Use st.write instead of st.dataframe for better style rendering
    st.write(styled_df)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="player-info-section">', unsafe_allow_html=True)
    if st.session_state.selected_player:
        player_data = filtered_top_scorers[filtered_top_scorers['Player'] == st.session_state.selected_player].iloc[0]
        st.subheader(f"Selected Player: {st.session_state.selected_player}")
        image_path = f"player_images/{st.session_state.selected_player.lower().replace(' ', '_')}.jpg"
        if os.path.exists(image_path):
            st.image(image_path, caption=st.session_state.selected_player, width=200)
        else:
            st.image("placeholder.jpg", caption="Player image not available", width=200)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="player-stats-section">', unsafe_allow_html=True)
    if st.session_state.selected_player:
        st.subheader("Player Stats")
        st.write(f"Team: {player_data['Team']}")
        st.write(f"Position: {player_data['Pos']}")
        st.write(f"Points per Game: {player_data['PTS']:.1f}")
        st.write(f"Assists per Game: {player_data['AST']:.1f}")
        st.write(f"Rebounds per Game: {player_data['TRB']:.1f}")
        # Add more stats as needed
    st.markdown('</div>', unsafe_allow_html=True)

# Display filtered data
st.subheader("Filtered Player Data")
st.markdown(f"**🔍 Dataset: {filtered_data.shape[0]} rows and {filtered_data.shape[1]} columns.**")
st.dataframe(filtered_data.style.background_gradient(cmap='coolwarm', subset=['PTS', 'AST', 'TRB']))

# CSV download function
def get_csv_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="wnba_stats.csv">Download CSV</a>'

st.markdown(get_csv_download_link(filtered_data), unsafe_allow_html=True)

# After displaying the data and visualizations
if st.button("Generate AI Insights"):
    with st.spinner("Generating insights..."):
        top_players = filtered_data.nlargest(10, 'PTS')
        key_stats = ['Player', 'Team', 'Pos', 'PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%', '3P%']
        summary_data = top_players[key_stats]
        
        insights = generate_insights(filtered_data)
        st.subheader("AI-Generated Insights")
        st.markdown(insights)
        
        st.warning("Please note: These insights are AI-generated based on the provided data. Always verify important information.")

# Add the sticky footer
st.markdown(
    """
    <div class="sticky-footer">
        Made with ❤️ w/ Cloudflare Workers AI in SF 
    </div>
    """,
    unsafe_allow_html=True
)

# # Add a chat interface for questions
# st.subheader("Ask about the data")
# user_question = st.text_input("Enter your question about the WNBA data:")

# if user_question:
#     prompt = f"""
#     Given the WNBA player statistics for the {selected_season} season, answer the following question:
#     {user_question}
    
#     Use the following data to inform your answer:
#     {filtered_data.to_string()}
#     """
    
#     with st.spinner("Generating answer..."):
#         response = requests.post(
#             f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/meta/llama-3.2-11b-vision-instruct",
#             headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
#             json={
#                 "messages": [
#                     {"role": "system", "content": "You are a helpful assistant that answers questions about WNBA player statistics."},
#                     {"role": "user", "content": prompt}
#                 ]
#             }
#         )
#         result = response.json()
#         answer = result['result']['response']
#         st.markdown(f"**Question:** {user_question}")
#         st.markdown(f"**Answer:** {answer}")
