import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import os
import time

# Set seaborn style for visualizations
sns.set(style="whitegrid")

# Define major players for each franchise (30 teams, 3 players each)
major_players = {
    "Boston Celtics": ["Jayson Tatum", "Jaylen Brown", "Isaiah Thomas"],
    "Cleveland Cavaliers": ["LeBron James", "Kyrie Irving", "Kevin Love"],
    "Denver Nuggets": ["Nikola Jokic", "Jamal Murray", "Carmelo Anthony"],
    "Golden State Warriors": ["Stephen Curry", "Klay Thompson", "Draymond Green"],
    "Houston Rockets": ["James Harden", "Chris Paul", "Russell Westbrook"],
    "LA Clippers": ["Kawhi Leonard", "Paul George", "Chris Paul"],
    "Los Angeles Lakers": ["LeBron James", "Anthony Davis", "Kobe Bryant"],
    "Miami Heat": ["Jimmy Butler", "Dwyane Wade", "Bam Adebayo"],
    "Milwaukee Bucks": ["Giannis Antetokounmpo", "Khris Middleton", "Jrue Holiday"],
    "Minnesota Timberwolves": ["Karl-Anthony Towns", "Anthony Edwards", "Rudy Gobert"],
    "New Orleans Pelicans": ["Anthony Davis", "Zion Williamson", "DeMarcus Cousins"],
    "Philadelphia 76ers": ["Joel Embiid", "Ben Simmons", "James Harden"],
    "Toronto Raptors": ["Kyle Lowry", "DeMar DeRozan", "Pascal Siakam"],
    "Utah Jazz": ["Donovan Mitchell", "Rudy Gobert", "Gordon Hayward"],
    "Washington Wizards": ["John Wall", "Bradley Beal", "Russell Westbrook"]
}

# Define seasons (2014-15 to 2023-24)
seasons = [
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"
]

# Initialize an empty list to store DataFrames
all_data = []

# Base directory for saving files
BASE_DIR = r"C:\Users\ibohn\basketstat-iq\basketstat-iq"

# Loop through each team and its major players
for team, player_names in major_players.items():
    for player_name in player_names:
        # Get player ID
        try:
            player_dict = players.find_players_by_full_name(player_name)[0]
            player_id = player_dict['id']
        except IndexError:
            print(f"Player {player_name} not found. Skipping.")
            continue

        for season in seasons:
            print(f"Collecting data for {player_name} ({team}) - Season {season}")
            try:
                # Fetch game logs
                gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
                df = gamelog.get_data_frames()[0]

                # Skip if no data is returned
                if df.empty:
                    print(f"No data for {player_name} - Season {season}. Skipping.")
                    continue

                # Add player name, team, and season columns
                df['PLAYER_NAME'] = player_name
                df['TEAM'] = team
                df['SEASON'] = season

                # Clean and prepare data
                df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
                df = df.sort_values('GAME_DATE')

                # Add moving averages for points, rebounds, assists, FG_PCT, PLUS_MINUS, and minutes
                df['POINTS_MOVING_AVG'] = df['PTS'].rolling(window=5).mean()
                df['REBOUNDS_MOVING_AVG'] = df['REB'].rolling(window=5).mean()
                df['ASSISTS_MOVING_AVG'] = df['AST'].rolling(window=5).mean()
                df['FG_PCT_MOVING_AVG'] = df['FG_PCT'].rolling(window=5).mean()
                df['PLUS_MINUS_MOVING_AVG'] = df['PLUS_MINUS'].rolling(window=5).mean()
                df['MIN_MOVING_AVG'] = df['MIN'].rolling(window=5).mean()

                # Add back-to-back indicator
                df['BACK_TO_BACK'] = df['GAME_DATE'].diff().dt.days <= 1

                # Add home/away indicator
                df['IS_HOME'] = df['MATCHUP'].str.contains('vs.')

                # Extract opponent team
                df['OPPONENT'] = df['MATCHUP'].str.split(' @ | vs. ').str[-1]

                # Calculate simplified Player Efficiency Rating (PER)
                df['SIMPLIFIED_PER'] = (df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK'] - df['TOV']) / df['MIN']
                df['SIMPLIFIED_PER'] = df['SIMPLIFIED_PER'].fillna(0)

                # Calculate offensive efficiency (points per possession)
                df['OFF_EFF'] = df['PTS'] / (df['FGA'] + 0.44 * df['FTA'] + df['TOV'])
                df['OFF_EFF'] = df['OFF_EFF'].fillna(0)

                # Calculate defensive rebound percentage (approximation)
                df['DEF_REB_PCT'] = df['DREB'] / df['REB']
                df['DEF_REB_PCT'] = df['DEF_REB_PCT'].fillna(0)

                # Save temporary file for this player/season
                os.makedirs(os.path.join(BASE_DIR, 'data', 'temp'), exist_ok=True)
                temp_file = os.path.join(BASE_DIR, 'data', 'temp', f"{player_name.replace(' ', '_').replace('.', '')}_{season}.csv")
                df.to_csv(temp_file, index=False)

                # Append to all_data
                all_data.append(df)

                # Plot points, rebounds, and assists
                plt.figure(figsize=(12, 8))
                sns.lineplot(x='GAME_DATE', y='PTS', data=df, label='Points')
                sns.lineplot(x='GAME_DATE', y='POINTS_MOVING_AVG', data=df, label='Points (5-Game Avg)')
                sns.lineplot(x='GAME_DATE', y='REB', data=df, label='Rebounds')
                sns.lineplot(x='GAME_DATE', y='REBOUNDS_MOVING_AVG', data=df, label='Rebounds (5-Game Avg)')
                sns.lineplot(x='GAME_DATE', y='AST', data=df, label='Assists')
                sns.lineplot(x='GAME_DATE', y='ASSISTS_MOVING_AVG', data=df, label='Assists (5-Game Avg)')
                plt.title(f"{player_name}'s Performance Metrics ({team}) - {season}")
                plt.xlabel('Game Date')
                plt.ylabel('Value')
                plt.xticks(rotation=45)
                plt.legend()
                plt.tight_layout()

                # Save plot
                os.makedirs(os.path.join(BASE_DIR, 'visuals'), exist_ok=True)
                safe_player_name = player_name.replace(" ", "_").replace(".", "")
                plt.savefig(os.path.join(BASE_DIR, 'visuals', f"{safe_player_name}_{season}_performance_metrics.png"))
                plt.close()

                # Pause to avoid API rate limits
                time.sleep(1.5)

            except Exception as e:
                print(f"Error fetching data for {player_name} ({team}) - Season {season}: {e}")
                continue

# Combine all data into a single DataFrame
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)

    # Save combined data
    os.makedirs(os.path.join(BASE_DIR, 'data', 'cleaned'), exist_ok=True)
    combined_df.to_csv(os.path.join(BASE_DIR, 'data', 'cleaned', 'combined_player_stats.csv'), index=False)
    print(f"Data saved to {os.path.join(BASE_DIR, 'data', 'cleaned', 'combined_player_stats.csv')}")
else:
    print("No data collected.")