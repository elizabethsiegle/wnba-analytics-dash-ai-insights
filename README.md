# WNBA Analytics Dashboard with AI Insights

## Overview

This project is an interactive WNBA (Women's National Basketball Association) analytics dashboard powered by Streamlit, LangChain, and Cloudflare Workers AI. It provides comprehensive statistics, player comparisons, and AI-driven insights about WNBA teams and players.

## Features

- Player statistics data visualizations
- Player vs Player comparisons
- Interactive map showing WNBA team locations
- AI-powered chatbot for WNBA-related queries powered by Cloudflare Workers AI and LangChain
- Data filtering and sorting capabilities

## Technologies Used

- Python
- Streamlit
- LangChain
- Cloudflare Workers AI
- Pandas for data manipulation
- Plotly for interactive charts
- Folium for map visualizations

## Getting Started

### Prerequisites

- Python 3.7+
- Cloudflare account--you'll need an account ID and Workers AI API token to run the app locally

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/elizabethsiegle/wnba-analytics-dash-ai-insights.git
   cd wnba-analytics-dash-ai-insights
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root and add your Cloudflare credentials:
   ```
   CF_ACCOUNT_ID=your_account_id
   CF_AUTH_TOKEN=your_api_token
   ```

### Running the App

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to whatever port it says the app is running on

## Usage

- Use the sidebar to filter data and select different views
- Interact with the charts to explore WNBA statistics
- Use the AI chatbot to ask questions about WNBA teams and players