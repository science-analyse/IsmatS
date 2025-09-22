import chess.pgn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

def parse_pgn_file(filename):
    """Parse a PGN file and extract game data"""
    games = []

    with open(filename, 'r', encoding='utf-8') as f:
        while True:
            try:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                headers = game.headers

                # Extract account name from headers
                account = None
                if 'White' in headers and 'Black' in headers:
                    if headers['White'] in ['IsmatS', 'Cassiny']:
                        account = headers['White']
                    elif headers['Black'] in ['IsmatS', 'Cassiny']:
                        account = headers['Black']

                # Determine if player won, lost, or drew
                result = headers.get('Result', '')
                if account:
                    def safe_int(value, default=0):
                        try:
                            return int(value)
                        except (ValueError, TypeError):
                            return default

                    if account == headers.get('White'):
                        player_result = 'win' if result == '1-0' else 'loss' if result == '0-1' else 'draw'
                        player_color = 'white'
                        player_elo = safe_int(headers.get('WhiteElo', 0))
                        opponent_elo = safe_int(headers.get('BlackElo', 0))
                        rating_diff = safe_int(headers.get('WhiteRatingDiff', 0))
                    else:
                        player_result = 'win' if result == '0-1' else 'loss' if result == '1-0' else 'draw'
                        player_color = 'black'
                        player_elo = safe_int(headers.get('BlackElo', 0))
                        opponent_elo = safe_int(headers.get('WhiteElo', 0))
                        rating_diff = safe_int(headers.get('BlackRatingDiff', 0))
                else:
                    continue

                # Parse time control
                time_control = headers.get('TimeControl', '')
                if '+' in time_control:
                    base_time, increment = map(int, time_control.split('+'))
                else:
                    base_time = int(time_control) if time_control.isdigit() else 0
                    increment = 0

                # Determine game type based on time control
                total_time = base_time + increment * 40  # Rough estimate
                if total_time < 180:
                    game_type = 'bullet'
                elif total_time < 600:
                    game_type = 'blitz'
                elif total_time < 1800:
                    game_type = 'rapid'
                else:
                    game_type = 'classical'

                # Count moves
                moves = []
                board = game.board()
                for move in game.mainline_moves():
                    moves.append(move)
                    board.push(move)

                game_data = {
                    'account': account,
                    'date': headers.get('Date', ''),
                    'utc_date': headers.get('UTCDate', ''),
                    'utc_time': headers.get('UTCTime', ''),
                    'result': player_result,
                    'color': player_color,
                    'player_elo': player_elo,
                    'opponent_elo': opponent_elo,
                    'elo_diff': opponent_elo - player_elo,
                    'rating_change': rating_diff,
                    'opening': headers.get('Opening', ''),
                    'eco_code': headers.get('ECO', ''),
                    'termination': headers.get('Termination', ''),
                    'time_control': time_control,
                    'game_type': game_type,
                    'base_time': base_time,
                    'increment': increment,
                    'total_moves': len(moves),
                    'site': headers.get('Site', ''),
                    'event': headers.get('Event', '')
                }

                games.append(game_data)

            except Exception as e:
                continue

    return games

def analyze_chess_data(df):
    """Perform comprehensive analysis of chess data"""
    # Convert date strings to datetime
    df['datetime'] = pd.to_datetime(df['utc_date'] + ' ' + df['utc_time'])
    df['date_only'] = pd.to_datetime(df['date'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['month'] = df['datetime'].dt.month
    return df

def create_individual_charts(df):
    """Create individual charts - one chart per PNG file"""

    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

    # Chart 1: Win Rate by Account
    plt.figure(figsize=(10, 6))
    win_rates = df.groupby('account')['result'].apply(lambda x: (x == 'win').mean() * 100)
    bars = plt.bar(win_rates.index, win_rates.values, color=['#2ecc71', '#3498db'])
    plt.title('Win Rate by Account', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Account', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    for i, v in enumerate(win_rates.values):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/01_win_rate_by_account.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 2: Results Distribution
    plt.figure(figsize=(12, 6))
    result_counts = df.groupby(['account', 'result']).size().unstack(fill_value=0)
    ax = result_counts.plot(kind='bar', color=['#e74c3c', '#f39c12', '#2ecc71'], figsize=(12, 6))
    plt.title('Win/Loss/Draw Distribution by Account', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Account', fontsize=12)
    plt.ylabel('Number of Games', fontsize=12)
    plt.legend(title='Result', fontsize=11)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/02_results_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 3: Rating Evolution - IsmatS
    ismats_data = df[df['account'] == 'IsmatS'].sort_values('datetime')
    if len(ismats_data) > 0:
        plt.figure(figsize=(14, 8))
        ismats_data = ismats_data.copy()
        ismats_data['cumulative_rating'] = ismats_data['player_elo'].iloc[0] + ismats_data['rating_change'].cumsum()
        plt.plot(ismats_data['datetime'], ismats_data['cumulative_rating'], linewidth=2, color='#3498db')

        # Add trend line
        x_numeric = np.arange(len(ismats_data))
        z = np.polyfit(x_numeric, ismats_data['cumulative_rating'], 1)
        p = np.poly1d(z)
        plt.plot(ismats_data['datetime'], p(x_numeric), "--", alpha=0.8, color='#e74c3c',
                label=f'Trend (slope: {z[0]:.1f})')

        plt.title('IsmatS - Rating Evolution Over Time', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Rating', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig('assets/03_ismats_rating_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Chart 4: Rating Evolution - Cassiny
    cassiny_data = df[df['account'] == 'Cassiny'].sort_values('datetime')
    if len(cassiny_data) > 0:
        plt.figure(figsize=(14, 8))
        cassiny_data = cassiny_data.copy()
        cassiny_data['cumulative_rating'] = cassiny_data['player_elo'].iloc[0] + cassiny_data['rating_change'].cumsum()
        plt.plot(cassiny_data['datetime'], cassiny_data['cumulative_rating'], linewidth=2, color='#9b59b6')

        # Add trend line
        x_numeric = np.arange(len(cassiny_data))
        z = np.polyfit(x_numeric, cassiny_data['cumulative_rating'], 1)
        p = np.poly1d(z)
        plt.plot(cassiny_data['datetime'], p(x_numeric), "--", alpha=0.8, color='#e74c3c',
                label=f'Trend (slope: {z[0]:.1f})')

        plt.title('Cassiny - Rating Evolution Over Time', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Rating', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig('assets/04_cassiny_rating_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Chart 5: Performance by Game Type
    plt.figure(figsize=(12, 7))
    game_type_performance = df.groupby(['account', 'game_type'])['result'].apply(lambda x: (x == 'win').mean() * 100).unstack(fill_value=0)
    ax = game_type_performance.plot(kind='bar', color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'], figsize=(12, 7))
    plt.title('Win Rate by Game Type', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Account', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.legend(title='Game Type', fontsize=11)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/05_performance_by_game_type.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 6: Performance by Color
    plt.figure(figsize=(10, 6))
    color_performance = df.groupby(['account', 'color'])['result'].apply(lambda x: (x == 'win').mean() * 100).unstack(fill_value=0)
    ax = color_performance.plot(kind='bar', color=['#2c3e50', '#ecf0f1'], figsize=(10, 6))
    plt.title('Win Rate by Color', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Account', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.legend(title='Color', fontsize=11)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/06_performance_by_color.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 7: Performance by Hour of Day
    plt.figure(figsize=(14, 7))
    hourly_performance = df.groupby('hour')['result'].apply(lambda x: (x == 'win').mean() * 100)
    plt.plot(hourly_performance.index, hourly_performance.values, marker='o', color='#9b59b6', linewidth=3, markersize=8)
    plt.title('Win Rate by Hour of Day', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Hour', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24, 2))
    plt.tight_layout()
    plt.savefig('assets/07_performance_by_hour.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 8: Performance by Day of Week
    plt.figure(figsize=(12, 7))
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_performance = df.groupby('day_of_week')['result'].apply(lambda x: (x == 'win').mean() * 100).reindex(day_order)
    bars = plt.bar(daily_performance.index, daily_performance.values, color='#e67e22')
    plt.title('Win Rate by Day of Week', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/08_performance_by_day.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 9: Top 10 Most Played Openings
    plt.figure(figsize=(14, 8))
    top_openings = df['opening'].value_counts().head(10)
    bars = plt.barh(range(len(top_openings)), top_openings.values, color='#1abc9c')
    plt.yticks(range(len(top_openings)), top_openings.index)
    plt.title('Top 10 Most Played Openings', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Games', fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/09_top_openings.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 10: Best Performing Openings
    plt.figure(figsize=(14, 8))
    opening_performance = df.groupby('opening').agg({
        'result': lambda x: (x == 'win').mean() * 100,
        'opening': 'count'
    }).rename(columns={'result': 'win_rate', 'opening': 'game_count'})
    best_openings = opening_performance[opening_performance['game_count'] >= 10].nlargest(10, 'win_rate')
    bars = plt.barh(range(len(best_openings)), best_openings['win_rate'], color='#27ae60')
    plt.yticks(range(len(best_openings)), best_openings.index)
    plt.title('Best Performing Openings (≥10 games)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Win Rate (%)', fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/10_best_openings.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 11: Game Length Distribution
    plt.figure(figsize=(12, 7))
    plt.hist(df['total_moves'], bins=40, alpha=0.7, color='#34495e', edgecolor='black')
    plt.axvline(df['total_moves'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {df["total_moves"].mean():.1f} moves')
    plt.title('Game Length Distribution (Total Moves)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Moves', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/11_game_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 12: Termination Reasons
    plt.figure(figsize=(10, 10))
    termination_counts = df['termination'].value_counts().head(8)
    colors = plt.cm.Set3(np.linspace(0, 1, len(termination_counts)))
    plt.pie(termination_counts.values, labels=termination_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors)
    plt.title('Game Termination Reasons', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('assets/12_termination_reasons.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 13: Win Rate vs Rating Difference
    plt.figure(figsize=(12, 7))
    df['elo_diff_bucket'] = pd.cut(df['elo_diff'],
                                   bins=[-float('inf'), -200, -100, -50, 0, 50, 100, 200, float('inf')],
                                   labels=['<-200', '-200 to -100', '-100 to -50', '-50 to 0',
                                          '0 to 50', '50 to 100', '100 to 200', '>200'])
    elo_diff_performance = df.groupby('elo_diff_bucket')['result'].apply(lambda x: (x == 'win').mean() * 100)
    bars = plt.bar(range(len(elo_diff_performance)), elo_diff_performance.values, color='#f39c12')
    plt.xticks(range(len(elo_diff_performance)), elo_diff_performance.index, rotation=45)
    plt.title('Win Rate vs Rating Difference\n(Negative = Opponent Stronger)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Rating Difference', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/13_winrate_vs_rating_diff.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 14: Monthly Activity
    plt.figure(figsize=(16, 7))
    df['year_month'] = df['datetime'].dt.to_period('M')
    monthly_total = df.groupby('year_month').size()
    plt.plot(range(len(monthly_total)), monthly_total.values, marker='o', color='#e74c3c', linewidth=2, markersize=6)
    plt.title('Monthly Game Activity', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Number of Games', fontsize=12)
    plt.xticks(range(0, len(monthly_total), max(1, len(monthly_total)//10)),
               [str(monthly_total.index[i]) for i in range(0, len(monthly_total), max(1, len(monthly_total)//10))],
               rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/14_monthly_activity.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Creating individual chess analysis charts...")

    # Parse both PGN files
    print("Parsing IsmatS.pgn...")
    ismats_games = parse_pgn_file('IsmatS.pgn')

    print("Parsing Cassiny.pgn...")
    cassiny_games = parse_pgn_file('Cassiny.pgn')

    # Combine data
    all_games = ismats_games + cassiny_games
    df = pd.DataFrame(all_games)

    # Analyze data
    df = analyze_chess_data(df)

    # Create individual charts
    print("Creating individual charts...")
    create_individual_charts(df)

    print("\nIndividual charts created successfully!")
    print("Each chart is now in its own PNG file in the assets/ folder")

if __name__ == "__main__":
    main()