import chess.pgn
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt

# Function to extract game information
def extract_game_info(pgn_stream):
    games = []
    while True:
        game = chess.pgn.read_game(pgn_stream)
        if game is None:
            break
        # Extract game headers
        headers = game.headers
        moves = game.mainline_moves()
        move_list = [str(move) for move in moves]
        games.append({
            'Event': headers.get('Event'),
            'Site': headers.get('Site'),
            'Date': headers.get('Date'),
            'White': headers.get('White'),
            'Black': headers.get('Black'),
            'Result': headers.get('Result'),
            'WhiteElo': headers.get('WhiteElo'),
            'BlackElo': headers.get('BlackElo'),
            'WhiteRatingDiff': headers.get('WhiteRatingDiff'),
            'BlackRatingDiff': headers.get('BlackRatingDiff'),
            'Variant': headers.get('Variant'),
            'TimeControl': headers.get('TimeControl'),
            'ECO': headers.get('ECO'),
            'Opening': headers.get('Opening'),
            'Termination': headers.get('Termination'),
            'Moves': move_list
        })
    return games

# Load and parse PGN file
with open('games.pgn', 'r') as file:
    pgn_data = file.read()

pgn_stream = StringIO(pgn_data)
games_data = extract_game_info(pgn_stream)
games_df = pd.DataFrame(games_data)

# Convert date to datetime
games_df['Date'] = pd.to_datetime(games_df['Date'])

# Extract month and year from the date
games_df['YearMonth'] = games_df['Date'].dt.to_period('M')

# Convert Elo ratings to numeric, errors='coerce' will replace non-numeric values with NaN
games_df['WhiteElo'] = pd.to_numeric(games_df['WhiteElo'], errors='coerce')
games_df['BlackElo'] = pd.to_numeric(games_df['BlackElo'], errors='coerce')

# Overall performance
results = games_df['Result'].value_counts()

# Performance by color
white_wins = games_df[games_df['Result'] == '1-0'].shape[0]
black_wins = games_df[games_df['Result'] == '0-1'].shape[0]
draws = games_df[games_df['Result'] == '1/2-1/2'].shape[0]

# Rating changes over time
white_rating_change = games_df.groupby('Date')['WhiteElo'].mean()
black_rating_change = games_df.groupby('Date')['BlackElo'].mean()

# Performance differences by month and year
performance_by_month = games_df.groupby('YearMonth')['Result'].value_counts().unstack().fillna(0)

# Opening analysis
opening_stats = games_df['Opening'].value_counts()

# Win rate by opening
openings = games_df['Opening'].unique()
opening_analysis = []
for opening in openings:
    total_games = games_df[games_df['Opening'] == opening].shape[0]
    wins = games_df[(games_df['Opening'] == opening) & (games_df['Result'] == '1-0')].shape[0]
    losses = games_df[(games_df['Opening'] == opening) & (games_df['Result'] == '0-1')].shape[0]
    draws = games_df[(games_df['Opening'] == opening) & (games_df['Result'] == '1/2-1/2')].shape[0]
    opening_analysis.append({
        'Opening': opening,
        'Total Games': total_games,
        'Wins': wins,
        'Losses': losses,
        'Draws': draws
    })
opening_analysis_df = pd.DataFrame(opening_analysis).sort_values(by='Total Games', ascending=False).reset_index(drop=True)

# Time control analysis
time_control_stats = games_df['TimeControl'].value_counts()

# Win rate by player
player_analysis = []
players = pd.concat([games_df['White'], games_df['Black']]).unique()
for player in players:
    total_games = games_df[(games_df['White'] == player) | (games_df['Black'] == player)].shape[0]
    wins = games_df[(games_df['White'] == player) & (games_df['Result'] == '1-0')].shape[0] + games_df[(games_df['Black'] == player) & (games_df['Result'] == '0-1')].shape[0]
    losses = games_df[(games_df['White'] == player) & (games_df['Result'] == '0-1')].shape[0] + games_df[(games_df['Black'] == player) & (games_df['Result'] == '1-0')].shape[0]
    draws = games_df[((games_df['White'] == player) | (games_df['Black'] == player)) & (games_df['Result'] == '1/2-1/2')].shape[0]
    player_analysis.append({
        'Player': player,
        'Total Games': total_games,
        'Wins': wins,
        'Losses': losses,
        'Draws': draws
    })
player_analysis_df = pd.DataFrame(player_analysis).sort_values(by='Total Games', ascending=False).reset_index(drop=True)

# Plotting and saving plots
plt.figure(figsize=(10, 6))
results.plot(kind='bar', title='Game Results')
plt.xlabel('Result')
plt.ylabel('Frequency')
plt.savefig('results.png')
plt.close()

plt.figure(figsize=(10, 6))
white_rating_change.plot(label='White Elo', legend=True)
black_rating_change.plot(label='Black Elo', legend=True)
plt.title('Rating Changes Over Time')
plt.xlabel('Date')
plt.ylabel('Elo Rating')
plt.savefig('rating_changes.png')
plt.close()

plt.figure(figsize=(10, 6))
ax = opening_stats.head(10).plot(kind='bar', title='Top 10 Openings')
plt.xlabel('Opening')
plt.ylabel('Frequency')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_openings.png')
plt.close()

plt.figure(figsize=(10, 6))
time_control_stats.head(10).plot(kind='bar', title='Time Controls Used')
plt.xlabel('Time Control')
plt.ylabel('Frequency')
plt.savefig('time_controls.png')
plt.close()

plt.figure(figsize=(10, 6))
performance_by_month.plot(kind='line', marker='o')
plt.title('Performance Over Time')
plt.xlabel('Month-Year')
plt.ylabel('Number of Games')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('performance_over_time.png')
plt.close()

plt.figure(figsize=(10, 6))
player_analysis_df.head(10).plot(kind='bar', x='Player', y='Total Games', title='Top 10 Players by Total Games')
plt.xlabel('Player')
plt.ylabel('Total Games')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('top_players.png')
plt.close()

# Write results to an HTML file
with open('results.html', 'w') as f:
    f.write("<h1>Chess Games Analysis</h1>")
    f.write("<h2>Overall Performance</h2>")
    f.write(results.to_frame().to_html())
    f.write(f"<p>Total number of games: {len(games_df)}</p>")
    f.write(f"<p>Performance by Color: White Wins: {white_wins}, Black Wins: {black_wins}, Draws: {draws}</p>")
    f.write("<h2>Opening Analysis</h2>")
    f.write(opening_analysis_df.head(10).to_html(index=False))
    f.write("<h2>Performance Over Time</h2>")
    f.write(performance_by_month.to_html())
    f.write("<h2>Top Players by Total Games</h2>")
    f.write(player_analysis_df.head(10).to_html(index=False))

# Save dataframe to CSV for further analysis if needed
games_df.to_csv('parsed_lichess_games.csv', index=False)
