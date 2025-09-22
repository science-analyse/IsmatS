import chess.pgn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
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
                account = None
                if 'White' in headers and 'Black' in headers:
                    if headers['White'] in ['IsmatS', 'Cassiny']:
                        account = headers['White']
                    elif headers['Black'] in ['IsmatS', 'Cassiny']:
                        account = headers['Black']

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

                time_control = headers.get('TimeControl', '')
                if '+' in time_control:
                    base_time, increment = map(int, time_control.split('+'))
                else:
                    base_time = int(time_control) if time_control.isdigit() else 0
                    increment = 0

                total_time = base_time + increment * 40
                if total_time < 180:
                    game_type = 'bullet'
                elif total_time < 600:
                    game_type = 'blitz'
                elif total_time < 1800:
                    game_type = 'rapid'
                else:
                    game_type = 'classical'

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

def analyze_combined_data(df):
    """Analyze as combined dataset"""
    df['datetime'] = pd.to_datetime(df['utc_date'] + ' ' + df['utc_time'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year

    def get_time_period(hour):
        if 5 <= hour < 9: return 'Early Morning'
        elif 9 <= hour < 12: return 'Late Morning'
        elif 12 <= hour < 14: return 'Lunch Time'
        elif 14 <= hour < 17: return 'Afternoon'
        elif 17 <= hour < 20: return 'Evening'
        elif 20 <= hour < 23: return 'Night'
        else: return 'Late Night'

    df['time_period'] = df['hour'].apply(get_time_period)
    df['weekend_weekday'] = df['day_of_week'].apply(lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday')

    # Calculate time since last game
    df_sorted = df.sort_values(['account', 'datetime'])
    df_sorted['time_since_last_game'] = df_sorted.groupby('account')['datetime'].diff()
    df_sorted['time_since_last_game_hours'] = df_sorted['time_since_last_game'].dt.total_seconds() / 3600

    return df_sorted

def create_remaining_charts(df):
    """Create remaining combined analysis charts"""

    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

    # Combined Chart 9: Your Game Length Distribution
    plt.figure(figsize=(12, 7))
    plt.hist(df['total_moves'], bins=40, alpha=0.7, color='#34495e', edgecolor='black')
    mean_moves = df['total_moves'].mean()
    plt.axvline(mean_moves, color='red', linestyle='--', linewidth=2,
               label=f'Your Average: {mean_moves:.1f} moves')
    plt.title('Your Game Length Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Moves per Game', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Add statistics text
    plt.text(0.7, 0.7, f'Total Games: {len(df):,}\nShortest: {df["total_moves"].min()} moves\nLongest: {df["total_moves"].max()} moves',
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

    plt.tight_layout()
    plt.savefig('assets/combined_09_game_length.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Combined Chart 10: Your Termination Reasons
    plt.figure(figsize=(10, 8))
    termination_counts = df['termination'].value_counts().head(8)
    colors = plt.cm.Set3(np.linspace(0, 1, len(termination_counts)))

    wedges, texts, autotexts = plt.pie(termination_counts.values, labels=termination_counts.index,
                                      autopct='%1.1f%%', startangle=90, colors=colors)

    plt.title('How Your Games End\n(Termination Reasons)', fontsize=16, fontweight='bold', pad=20)

    # Improve text readability
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    plt.tight_layout()
    plt.savefig('assets/combined_10_termination_reasons.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Combined Chart 11: Your Performance vs Rating Difference
    plt.figure(figsize=(12, 7))
    df['elo_diff_bucket'] = pd.cut(df['elo_diff'],
                                   bins=[-float('inf'), -200, -100, -50, 0, 50, 100, 200, float('inf')],
                                   labels=['Much Stronger\n(<-200)', 'Stronger\n(-200 to -100)',
                                          'Slightly Stronger\n(-100 to -50)', 'Similar\n(-50 to 0)',
                                          'Similar\n(0 to 50)', 'Slightly Weaker\n(50 to 100)',
                                          'Weaker\n(100 to 200)', 'Much Weaker\n(>200)'])

    elo_diff_performance = df.groupby('elo_diff_bucket').agg({
        'result': [lambda x: (x == 'win').mean() * 100, 'count']
    }).round(1)
    elo_diff_performance.columns = ['win_rate', 'game_count']

    # Filter buckets with at least 50 games
    elo_diff_performance = elo_diff_performance[elo_diff_performance['game_count'] >= 50]

    x_pos = range(len(elo_diff_performance))
    bars = plt.bar(x_pos, elo_diff_performance['win_rate'], color='#f39c12')

    # Add game count labels
    for i, (idx, row) in enumerate(elo_diff_performance.iterrows()):
        plt.text(i, row['win_rate'] + 2, f"{int(row['game_count'])} games",
                ha='center', va='bottom', fontweight='bold', fontsize=9)
        plt.text(i, row['win_rate']/2, f"{row['win_rate']:.1f}%",
                ha='center', va='center', fontweight='bold', color='white', fontsize=11)

    plt.title('Your Win Rate vs Opponent Strength\n(Rating Difference)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Opponent Strength Relative to You', fontsize=12)
    plt.ylabel('Your Win Rate (%)', fontsize=12)
    plt.xticks(x_pos, elo_diff_performance.index, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('assets/combined_11_rating_difference.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Combined Chart 12: Your Activity Over Time
    plt.figure(figsize=(16, 8))
    df['year_month'] = df['datetime'].dt.to_period('M')
    monthly_activity = df.groupby('year_month').size()

    x_pos = range(len(monthly_activity))
    bars = plt.bar(x_pos, monthly_activity.values, color='#e74c3c', alpha=0.8)

    # Highlight recent activity surge
    recent_threshold = len(monthly_activity) - 12  # Last 12 months
    for i in range(recent_threshold, len(monthly_activity)):
        bars[i].set_color('#27ae60')

    plt.title('Your Chess Activity Over Time\n(Green = Recent Activity Surge)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Games per Month', fontsize=12)

    # Show only every nth label to avoid crowding
    step = max(1, len(monthly_activity) // 15)
    plt.xticks(range(0, len(monthly_activity), step),
               [str(monthly_activity.index[i]) for i in range(0, len(monthly_activity), step)],
               rotation=45)

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('assets/combined_12_activity_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Combined Chart 13: Your Color Performance
    plt.figure(figsize=(10, 6))
    color_stats = df.groupby('color').agg({
        'result': [lambda x: (x == 'win').mean() * 100, 'count']
    }).round(1)
    color_stats.columns = ['win_rate', 'game_count']

    x_pos = range(len(color_stats))
    bars = plt.bar(x_pos, color_stats['win_rate'], color=['#ecf0f1', '#2c3e50'])

    for i, (idx, row) in enumerate(color_stats.iterrows()):
        plt.text(i, row['win_rate'] + 1, f"{int(row['game_count'])} games",
                ha='center', va='bottom', fontweight='bold')
        plt.text(i, row['win_rate']/2, f"{row['win_rate']:.1f}%",
                ha='center', va='center', fontweight='bold',
                color='black' if idx == 'white' else 'white', fontsize=12)

    plt.title('Your Performance by Piece Color', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Piece Color', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.xticks(x_pos, ['White Pieces', 'Black Pieces'])
    plt.grid(True, alpha=0.3, axis='y')

    # Add first move advantage note
    white_wr = color_stats.loc['white', 'win_rate']
    black_wr = color_stats.loc['black', 'win_rate']
    advantage = white_wr - black_wr
    plt.text(0.5, max(white_wr, black_wr) + 3, f'First Move Advantage: +{advantage:.1f}%',
             ha='center', fontweight='bold', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    plt.tight_layout()
    plt.savefig('assets/combined_13_color_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Combined Chart 14: Your Rest Period Optimization
    if 'time_since_last_game_hours' in df.columns:
        plt.figure(figsize=(12, 7))

        # Filter reasonable time gaps (between 1 minute and 7 days)
        time_gap_data = df[(df['time_since_last_game_hours'] >= 0.017) &
                          (df['time_since_last_game_hours'] <= 168)].copy()

        # Categorize time gaps
        def categorize_time_gap(hours):
            if hours < 1:
                return '< 1 hour'
            elif hours < 6:
                return '1-6 hours'
            elif hours < 24:
                return '6-24 hours'
            elif hours < 72:
                return '1-3 days'
            else:
                return '3+ days'

        time_gap_data['time_gap_category'] = time_gap_data['time_since_last_game_hours'].apply(categorize_time_gap)

        gap_stats = time_gap_data.groupby('time_gap_category').agg({
            'result': [lambda x: (x == 'win').mean() * 100, 'count']
        }).round(1)
        gap_stats.columns = ['win_rate', 'game_count']

        # Reorder categories logically
        category_order = ['< 1 hour', '1-6 hours', '6-24 hours', '1-3 days', '3+ days']
        gap_stats = gap_stats.reindex([cat for cat in category_order if cat in gap_stats.index])

        x_pos = range(len(gap_stats))
        bars = plt.bar(x_pos, gap_stats['win_rate'], color='#3498db')

        for i, (idx, row) in enumerate(gap_stats.iterrows()):
            plt.text(i, row['win_rate'] + 1, f"{int(row['game_count'])}",
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
            plt.text(i, row['win_rate']/2, f"{row['win_rate']:.1f}%",
                    ha='center', va='center', fontweight='bold', color='white', fontsize=11)

        plt.title('Your Performance vs Rest Time Between Games', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Time Since Last Game', fontsize=12)
        plt.ylabel('Win Rate (%)', fontsize=12)
        plt.xticks(x_pos, gap_stats.index, rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        # Highlight optimal range
        optimal_idx = gap_stats['win_rate'].idxmax()
        optimal_pos = list(gap_stats.index).index(optimal_idx)
        plt.text(optimal_pos, gap_stats.loc[optimal_idx, 'win_rate'] + 3, 'OPTIMAL',
                ha='center', fontweight='bold', color='green', fontsize=12)

        plt.tight_layout()
        plt.savefig('assets/combined_14_rest_periods.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    print("Creating remaining combined analysis charts...")

    # Parse both PGN files
    print("Parsing IsmatS.pgn...")
    ismats_games = parse_pgn_file('IsmatS.pgn')

    print("Parsing Cassiny.pgn...")
    cassiny_games = parse_pgn_file('Cassiny.pgn')

    # Combine data
    all_games = ismats_games + cassiny_games
    df = pd.DataFrame(all_games)

    # Analyze combined data
    df = analyze_combined_data(df)

    # Create remaining combined charts
    print("Creating remaining combined analysis charts...")
    create_remaining_charts(df)

    print("\nRemaining combined analysis charts created successfully!")
    print("Generated 6 additional charts completing your unified analysis!")

if __name__ == "__main__":
    main()