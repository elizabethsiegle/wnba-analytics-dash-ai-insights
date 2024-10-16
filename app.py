import base64
from dotenv import load_dotenv
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from langchain_community.llms.cloudflare_workersai import CloudflareWorkersAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
import os
import pandas as pd
from pathlib import Path

import plotly.graph_objects as go
# import plotly.express as px
import requests
import streamlit as st
import time
import webcolors



load_dotenv()

# Cloudflare Workers AI setup
ACCOUNT_ID = os.getenv('CF_ACCOUNT_ID') # st.secrets["CF_ACCOUNT_ID"] # 
AUTH_TOKEN = os.getenv('CF_AUTH_TOKEN') # st.secrets["CF_AUTH_TOKEN"] # 

# Updated Page configuration
st.set_page_config(page_title="WNBA Player Analytics Dashboard, AI Insights, && AI Assistant", page_icon="🏀", layout="wide")

# Enhanced Custom CSS with gradient background, hover effects, and sticky footer
st.markdown("""
<style>
    .hover-link {
        color: #1E90FF;  /* Initial color - dodger blue */
        text-decoration: none;
        transition: color 0.3s ease;
    }
    .hover-link:hover {
        color: #FF4500;  /* Hover color - orange red */
        text-decoration: underline;
    }
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: #2C3E50;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"]:nth-of-type(2n) {
        background-color: #34495E;
    }
    
    .main-content {
        display: flex;
        justify-content: space-between;
    }
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
    .chart-section, .stats-section, .chatbot-section, .filtered-data-section, .player-comparison-section {
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
    .player-comparison-section {
        background-color: rgba(220, 20, 60, 0.2);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .chatbot-section {
        background-color: rgba(255, 69, 0, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .map-section {
        background-color: rgba(144, 238, 144, 0.2);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    
    }

    .map-section .stfolium {
        width: 100%;
        height: 400px;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
            
    .filtered-data-section {
        background-color: #4C51BF;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        border: 2px solid #60A5FA;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* AI Insights Section Styling */
    .ai-insights-section {
        background-color: #4B5563;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        border: 2px solid #60A5FA;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Custom button styling */
    button {
        display: inline-block;
        width: 500%;
        padding: 12px 20px;
        font-size: 40px;
        font-weight: bold;
        color: white;
        background-color: #3B82F6;
        border: none;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(59, 130, 246, 0.5);  /* Enhanced shadow */
        text-align: center;
        text-decoration: none;
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
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
            Explore comprehensive WNBA player statistics with this interactive dashboard, map, and AI Insights and chatbot powered by LangChain and Cloudflare Workers AI.
            <br><strong>Data Source:</strong> <a href="https://www.basketball-reference.com/" target="_blank" class="hover-link">Basketball-reference.com</a>
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
}

# Function to get team abbreviation from full name
def get_team_abbr(team_name):
    abbr_map = {
        'Atlanta Dream': 'ATL', 'Chicago Sky': 'CHI', 'Connecticut Sun': 'CON',
        'Dallas Wings': 'DAL', 'Indiana Fever': 'IND', 'Las Vegas Aces': 'LVA',
        'Los Angeles Sparks': 'LAS', 'Minnesota Lynx': 'MIN', 'New York Liberty': 'NYL',
        'Phoenix Mercury': 'PHO', 'Seattle Storm': 'SEA', 'Washington Mystics': 'WAS'
    }
    return abbr_map.get(team_name, '')
selected_teams = st.sidebar.multiselect('Team', teams, teams)
# Apply the colors to the teams in the multiselect
colorize_multiselect_options(teams, team_colors)

positions = ['C', 'F', 'G', 'F-G', 'C-F']
selected_positions = st.sidebar.multiselect('Position', positions, positions)

# First, remove the duplicate entry for Celeste Taylor
player_data = player_data[~((player_data['Player'] == 'Celeste Taylor') & (player_data['Team'] == 'Connecticut Sun'))]

# Then apply the data filters
filtered_data = player_data[
    (player_data.Team.isin(selected_teams)) & 
    (player_data.Pos.isin(selected_positions))
].sort_values(by='PTS', ascending=False)  # Sort by points


# Move the pie chart above the displayed data
st.subheader("Top 5 Scorers (Points per Game)")

def closest_color(rgb):
    colors = {
        'red': (255, 0, 0), 'blue': (0, 0, 255), 'green': (0, 128, 0),
        'purple': (128, 0, 128), 'orange': (255, 165, 0),
        'pink': (255, 192, 203), 'black': (0, 0, 0), 'white': (255, 255, 255),
        'gray': (128, 128, 128), 'lightblue': (173, 216, 230), 'lightgreen': (144, 238, 144),
        'lightred': (255, 102, 102), 'beige': (245, 245, 220), 'darkblue': (0, 0, 139),
        'darkgreen': (0, 100, 0), 'darkpurple': (48, 25, 52), 'cadetblue': (95, 158, 160),
        'darkred': (139, 0, 0), 'lightgray': (211, 211, 211)
    }
    return min(colors, key=lambda color: sum((a-b)**2 for a, b in zip(colors[color], rgb)))

# Fallback coordinates for WNBA teams
fallback_coordinates = {
    'Atlanta Dream': (33.7490, -84.3880),
    'Chicago Sky': (41.8781, -87.6298),
    'Connecticut Sun': (41.4901, -72.0992),
    'Dallas Wings': (32.7355, -97.1080),
    'Indiana Fever': (39.7684, -86.1581),
    'Las Vegas Aces': (36.1699, -115.1398),
    'Los Angeles Sparks': (34.0522, -118.2437),
    'Minnesota Lynx': (44.9778, -93.2650),
    'New York Liberty': (40.6782, -73.9442),
    'Phoenix Mercury': (33.4484, -112.0740),
    'Seattle Storm': (47.6062, -122.3321),
    'Washington Mystics': (38.9072, -77.0369)
}

def geocode_with_retry(geolocator, city, max_retries=3):
    for _ in range(max_retries):
        try:
            return geolocator.geocode(city)
        except (GeocoderTimedOut, GeocoderServiceError):
            time.sleep(1)
    return None

@st.cache_data
def create_wnba_map():
    # WNBA teams, their locations, and home page URLs
    wnba_teams = {
        'Atlanta Dream': ('Atlanta, GA', 'https://dream.wnba.com/'),
        'Chicago Sky': ('Chicago, IL', 'https://sky.wnba.com/'),
        'Connecticut Sun': ('Uncasville, CT', 'https://sun.wnba.com/'),
        'Dallas Wings': ('Arlington, TX', 'https://wings.wnba.com/'),
        'Indiana Fever': ('Indianapolis, IN', 'https://fever.wnba.com/'),
        'Las Vegas Aces': ('Las Vegas, NV', 'https://aces.wnba.com/'),
        'Los Angeles Sparks': ('Los Angeles, CA', 'https://sparks.wnba.com/'),
        'Minnesota Lynx': ('Minneapolis, MN', 'https://lynx.wnba.com/'),
        'New York Liberty': ('Brooklyn, NY', 'https://liberty.wnba.com/'),
        'Phoenix Mercury': ('Phoenix, AZ', 'https://mercury.wnba.com/'),
        'Seattle Storm': ('Seattle, WA', 'https://storm.wnba.com/'),
        'Washington Mystics': ('Washington, D.C.', 'https://mystics.wnba.com/')
    }

    # Create a map centered on the United States
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

    # Geocoding to get coordinates
    geolocator = Nominatim(user_agent="wnba_app")
    # Team name to abbreviation mapping
    team_abbr = {
        'Atlanta Dream': 'ATL', 'Chicago Sky': 'CHI', 'Connecticut Sun': 'CON',
        'Dallas Wings': 'DAL', 'Indiana Fever': 'IND', 'Las Vegas Aces': 'LVA',
        'Los Angeles Sparks': 'LAS', 'Minnesota Lynx': 'MIN', 'New York Liberty': 'NYL',
        'Phoenix Mercury': 'PHO', 'Seattle Storm': 'SEA', 'Washington Mystics': 'WAS'
    }

    # Add markers for each team
    for team, (city, url) in wnba_teams.items():
        try:
            location = geocode_with_retry(geolocator, city)
            if location is None:
                # Use fallback coordinates if geocoding fails
                lat, lon = fallback_coordinates[team]
            else:
                lat, lon = location.latitude, location.longitude
            # Get team abbreviation and color
            abbr = team_abbr.get(team, 'ATL')  # Default to ATL if not found
            hex_color = team_colors.get(abbr, '#000000')  # Default to black if color not found
            rgb = webcolors.hex_to_rgb(hex_color)
            closest_folium_color = closest_color(rgb)
            
            # Create popup HTML with team info and link
            popup_html = f"""
            <b>{team}</b><br>
            {city}<br>
            <a href="{url}" target="_blank">Visit Team Website</a>
            """
            
            folium.Marker(
                [lat, lon],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=team,
                icon=folium.Icon(color=closest_folium_color, icon='basketball', prefix='fa')
            ).add_to(m)
        except Exception as e:
            st.warning(f"Couldn't add marker for {team}: {str(e)}")

    return m

# Initialize session state for selected player
if 'selected_player' not in st.session_state:
    st.session_state.selected_player = None

# Function to update selected player
def update_selected_player(player):
    st.session_state.selected_player = player

def clean_percentage(value):
    if isinstance(value, str):
        return float(value.strip('%')) / 100
    return value

stat_options = {
    "Points": "PTS",
    "Assists": "AST",
    "Rebounds": "TRB",
    "Steals": "STL",
    "Blocks": "BLK",
    "Field Goal %": "FG%",
    "3-Point %": "3P%"
}

# Create a 2x2 grid layout
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    st.markdown('<div class="chart-section">', unsafe_allow_html=True)
    st.markdown("Player Statistics")
    # Add dropdown for stat selection
    selected_stat = st.selectbox("Select Statistic", list(stat_options.keys()))
    stat_column = stat_options[selected_stat]
    
    st.subheader(f"Top Players ({selected_stat})")
    chart_type = st.radio("Select chart type:", ("Pie Chart", "Bar Chart"))

    # Convert the selected statistic to numeric values
    filtered_data[stat_column] = pd.to_numeric(filtered_data[stat_column], errors='coerce')

    # Move the slider here, before creating top_players
    max_stat_value = filtered_data[stat_column].max()
    min_stat_value = st.slider(f"Minimum {selected_stat}", 
                               min_value=0.0, 
                               max_value=float(max_stat_value), 
                               value=0.0,
                               step=0.1)

    # Filter the data based on the slider value
    filtered_data = filtered_data[filtered_data[stat_column] >= min_stat_value]

    # Get top 5 players after filtering
    top_players = filtered_data.nlargest(5, stat_column)
    
    hover_text = [f"{player}<br>Team: {team}<br>Position: {pos}<br>{selected_stat}: {value:.2f}" 
                  for player, team, pos, value in zip(top_players['Player'], top_players['Team'], top_players['Pos'], top_players[stat_column])]

    colors = [team_colors.get(team, '#000000') for team in top_players['Team']]

    if chart_type == "Bar Chart":
        fig = go.Figure(data=[go.Bar(
            x=top_players['Player'],
            y=top_players[stat_column],
            marker_color=colors,  # Use team colors for bars
            text=top_players[stat_column],
            textposition='auto',
        )])
        fig.update_layout(
            title=f"Top 5 Players - {selected_stat}",
            xaxis_title="Player",
            yaxis_title=selected_stat,
        )
    else:  # Pie Chart
        fig = go.Figure(data=[go.Pie(
            labels=top_players['Player'],
            values=top_players[stat_column],
            marker=dict(colors=colors),  # Use team colors for pie slices
            textposition='inside',
            textinfo='label+percent',
            insidetextorientation='radial',
            hole=0.3,
        )])
        fig.update_traces(textfont_size=12)
        fig.update_layout(
            title=f"Top 5 Players - {selected_stat} Distribution",
            showlegend=False,
        )

    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<div class="stats-section">', unsafe_allow_html=True)
    st.markdown("Quick Stats")
    if not top_players.empty:
        st.metric(f"Average {selected_stat}", f"{top_players[stat_column].mean():.2f}")
        highest_player = top_players.loc[top_players[stat_column].idxmax()]
        st.metric(f"Highest {selected_stat}", f"{highest_player['Player']} ({highest_player[stat_column]:.2f})")
        st.metric("Players Shown", f"{len(top_players)}")
    else:
        st.warning("No players match the current filter.")
    
    st.markdown(f"### Top {selected_stat}")
    # Ensure the selected stat is numeric and round to 2 decimal places
    top_players[stat_column] = pd.to_numeric(top_players[stat_column], errors='coerce').round(2)

    # Create a custom style function
    def color_team(val):
        color = team_colors.get(val, 'black')  # Default to black if team not found
        return f'color: {color}; font-weight: bold;'

    # Apply the style to the dataframe
    styled_df = top_players[["Player", stat_column, "Team"]].style\
        .set_properties(**{'background-color': '#f0f0f0'})\
        .map(color_team, subset=['Player', 'Team'])\
        .map(lambda x: 'color: black; font-weight: bold;', subset=[stat_column])\
        .format({stat_column: '{:.2f}'})

    # Use st.write instead of st.dataframe for better style rendering
    st.write(styled_df)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="ai-insights-section">', unsafe_allow_html=True)
    st.subheader("🤖 AI Insights")
    
    # Check if the button is clicked (you'll need to implement this logic)
    if st.button("Generate AI🧠 Insights"):
        with st.spinner("Generating insights..."):
            top_players = filtered_data.nlargest(10, 'PTS')
            key_stats = ['Player', 'Team', 'Pos', 'PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%', '3P%']
            summary_data = top_players[key_stats]
            
            insights = generate_insights(filtered_data)
            st.subheader("AI-Generated Insights")
            st.markdown(insights)
            
            st.warning("Please note: These insights are AI-generated based on the provided data. Always verify important information.")
        
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    # Display filtered data
    st.markdown('<div class="filtered-data-section">', unsafe_allow_html=True)
    st.subheader("Filtered Player Data")
    st.markdown(f"**🔍 Dataset: {filtered_data.shape[0]} rows and {filtered_data.shape[1]} columns.**")
    # Format numeric columns
    numeric_columns = filtered_data.select_dtypes(include=['float64', 'int64']).columns
    formatter = {}
    for col in numeric_columns:
        if col == 'PTS':
            formatter[col] = '{:.0f}'.format
        else:
            formatter[col] = '{:.2f}'.format

        # Apply the formatting and style the dataframe
        styled_data = (filtered_data.style
                    .format(formatter)
                    .background_gradient(cmap='coolwarm', subset=['PTS', 'AST', 'TRB'])
                    .set_properties(**{'text-align': 'center'})
                    .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
        )
        # Display the styled dataframe
        st.dataframe(styled_data, height=400)
        # CSV download function
        def get_csv_download_link(df):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            return f'<a href="data:file/csv;base64,{b64}" download="wnba_stats.csv">Download CSV</a>'

        st.markdown(get_csv_download_link(filtered_data), unsafe_allow_html=True)

# Create two main columns for the sections
col1, col2 = st.columns(2)    
with col1:
    st.markdown('<div class="player-comparison-section">', unsafe_allow_html=True)
    st.subheader("🏀 Player Comparison (players must have played in the same season)")

    # Allow users to select players to compare
    players = player_data['Player'].unique()

    # Find the indices of Caitlin Clark and Angel Reese
    caitlin_index = players.tolist().index('Caitlin Clark') if 'Caitlin Clark' in players else 0
    angel_index = players.tolist().index('Angel Reese') if 'Angel Reese' in players else 0

    player1 = st.selectbox("Select first player", players,index=caitlin_index, key='player1')
    player2 = st.selectbox("Select second player", players, index=angel_index, key='player2')

    def normalize(value, min_value, max_value):
        try:
            value = float(value)
            return 100 * (value - min_value) / (max_value - min_value) if max_value > min_value else 50
        except (ValueError, TypeError):
            return 0  # or some default value for non-numeric entries

    if player1 and player2:
        # Get data for selected players
        stats1 = player_data[player_data['Player'] == player1].iloc[0]
        stats2 = player_data[player_data['Player'] == player2].iloc[0]

        # Select stats to compare
        stats_to_compare = ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'FG%', '3P%', 'FT%']
        # Convert columns to numeric, replacing non-numeric values with NaN
        for stat in stats_to_compare:
            player_data[stat] = pd.to_numeric(player_data[stat], errors='coerce')

        normalized_stats = {}
        for stat in stats_to_compare:
            min_val = player_data[stat].min()
            max_val = player_data[stat].max()
            normalized_stats[stat] = [
                normalize(stats1[stat], min_val, max_val),
                normalize(stats2[stat], min_val, max_val)
            ]

        # Create a radar chart
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=[normalized_stats[stat][0] for stat in stats_to_compare],
            theta=stats_to_compare,
            fill='toself',
            name=player1
        ))
        fig.add_trace(go.Scatterpolar(
            r=[normalized_stats[stat][1] for stat in stats_to_compare],
            theta=stats_to_compare,
            fill='toself',
            name=player2
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            legend=dict(
                font=dict(size=16),  # Increase font size
                itemsizing='constant',  # Make legend items a constant size
                itemwidth=30,  # Adjust item width
                yanchor="top",  # Anchor to the top
                y=0.99,  # Position at the top
                xanchor="right",  # Anchor to the right
                x=0.99,  # Position at the left
                bgcolor="rgba(255, 255, 255, 0.5)",  # Semi-transparent background
                bordercolor="Black",  # Border color
                borderwidth=2,  # Border width
            ),
            title=dict(
                text=f"{player1} vs {player2} Comparison",
                font=dict(size=24)  # Increase title font size
            ),
            width=700,  # Adjust as needed
            height=700  # Adjust as needed
        )
        # Create three columns with the middle one being wider
        left_col, middle_col, right_col = st.columns([1, 3, 1])

        # Use the middle column to display the chart
        with middle_col:
            st.plotly_chart(fig, use_container_width=True)

        # Display a table with the exact values
        comparison_df = pd.DataFrame({
            'Stat': stats_to_compare,
            player1: [stats1[stat] for stat in stats_to_compare],
            player2: [stats2[stat] for stat in stats_to_compare]
        })
        st.table(comparison_df)

with col2:
    st.markdown('<div class="chatbot-section">', unsafe_allow_html=True)
    st.subheader("🏀 Chat💬 w/ WNBA AI Assistant powered by LangChain && Cloudflare Workers AI🤖")

    # Add a loading message
    chat_loading = st.empty()
    chat_loading.info("Chat is initializing... This may take a few moments.")

    # Initialize the LLM and conversation chain
    @st.cache_resource
    def initialize_chat(filtered_data: pd.DataFrame):
        llm = CloudflareWorkersAI(
            account_id=ACCOUNT_ID,
            api_token=AUTH_TOKEN,
            model="@cf/meta/llama-2-7b-chat-int8"
        )
        # Convert filtered_data to a string representation
        data_context = filtered_data.to_string()

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable assistant specializing in WNBA statistics, players, and teams. 
            Provide accurate and helpful information about the WNBA.
            Here's the current WNBA data you have access to:
            {data_context}
            Use this data to answer questions, but don't mention the data directly unless asked."""),
            ("human", "{input}"),
            ("ai", "{agent_scratchpad}")
        ])

        memory = ConversationBufferMemory(return_messages=True, output_key="agent_scratchpad")
        def get_chat_history(inputs):
            return memory.chat_memory.messages

        chain = (
            RunnablePassthrough.assign(
                agent_scratchpad=get_chat_history,
                data_context=lambda _: data_context[:100] + "..." # Truncate for brevity 
            )
            | prompt
            | llm
        )

        return chain, memory, data_context

    # Initialize the chat
    chain, memory, data_context = initialize_chat(filtered_data)

    # Remove the loading message
    chat_loading.empty()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if user_input := st.chat_input("Ask me anything about WNBA stats, players, or teams!"):
        # Display user message in chat message container
        st.chat_message("user").markdown(user_input)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get AI response
        with st.spinner("Thinking..."):
            try:
                response = chain.invoke({
                    "input": user_input,
                    "data_context": data_context
                })
                
                if not response or response.strip() == "":
                    response = "I apologize, but I couldn't generate a response. This could be due to an issue with the AI model or the input. Please try asking your question in a different way or try again later."
            except Exception as e:
                response = f"An error occurred: {str(e)}"
                st.error(f"Debug: Error details: {e}")
        
        # After getting the response from the model
        if isinstance(response, list) and len(response) > 0 and hasattr(response[0], 'content'):
            response_text = response[0].content
        elif isinstance(response, dict) and 'content' in response:
            response_text = response['content']
        elif isinstance(response, str):
            response_text = response
        else:
            response_text = str(response)


        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response_text)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        
        # Update memory
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(response_text)


        # Add some styling to make the chat interface look better
        st.markdown("""
        <style>
        .stChatFloatingInputContainer {
            bottom: 20px;
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown('<div class="map-section">', unsafe_allow_html=True)
    st.subheader("🗺️📍WNBA Team Locations")

    # Custom CSS to center the map
    st.markdown("""
    <style>
    .map-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Debug: Check if create_wnba_map() is working
    try:
        wnba_map = create_wnba_map()
        # Removed the success message to save space
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")

    # Try to display the map
    try:
        # Wrap the map in a centered div
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        st_folium(wnba_map, width=700, height=500)  # Increased size
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying map: {str(e)}")

    # Add a caption below the map
    st.caption("Click on a marker to learn more about each team.")

    st.markdown('</div>', unsafe_allow_html=True)


 # Add custom CSS for the sticky footer
st.markdown(
    """
    <style>
    .reportview-container {
        flex-direction: column;
    }
    .main .block-container {
        padding-bottom: 70px;
    }
    .sticky-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 50px;
        background-color: #262730;
        color: #ffffff;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 14px;
        z-index: 999;
    }
    .sticky-footer a {
        color: #4da6ff;
        text-decoration: none;
        margin-left: 5px;
    }
    .sticky-footer a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add the sticky footer at the end of your app
st.markdown(
    """
    <div class="sticky-footer">
        Made with ❤️ w/ Cloudflare Workers AI in SF 
        <a href="https://github.com/elizabethsiegle/wnba-analytics-dash-ai-insights" target="_blank">Code here on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)