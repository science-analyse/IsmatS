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
                print(f"Error parsing game: {e}")
                continue

    return games

def analyze_chess_data(df):
    """Perform comprehensive analysis of chess data"""

    print(f"Total games analyzed: {len(df)}")
    print(f"Games by account: {df['account'].value_counts().to_dict()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Convert date strings to datetime
    df['datetime'] = pd.to_datetime(df['utc_date'] + ' ' + df['utc_time'])
    df['date_only'] = pd.to_datetime(df['date'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['month'] = df['datetime'].dt.month

    return df

def create_charts(df):
    """Create comprehensive charts for chess analysis"""

    # Set up the plotting style
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

    # Chart 1: Win/Loss/Draw Distribution by Account
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Overall results
    result_counts = df.groupby(['account', 'result']).size().unstack(fill_value=0)
    result_counts.plot(kind='bar', ax=ax1, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax1.set_title('Win/Loss/Draw Distribution by Account', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Account')
    ax1.set_ylabel('Number of Games')
    ax1.legend(title='Result')
    ax1.tick_params(axis='x', rotation=45)

    # Win rates
    win_rates = df.groupby('account')['result'].apply(lambda x: (x == 'win').mean() * 100)
    win_rates.plot(kind='bar', ax=ax2, color='#2ecc71')
    ax2.set_title('Win Rate by Account', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Account')
    ax2.set_ylabel('Win Rate (%)')
    ax2.tick_params(axis='x', rotation=45)

    for i, v in enumerate(win_rates.values):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('assets/chart1_results_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 2: Rating Evolution Over Time
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    for i, account in enumerate(df['account'].unique()):
        account_data = df[df['account'] == account].sort_values('datetime')

        # Calculate cumulative rating change
        account_data = account_data.copy()
        account_data['cumulative_rating'] = account_data['player_elo'].iloc[0] + account_data['rating_change'].cumsum()

        axes[i].plot(account_data['datetime'], account_data['cumulative_rating'],
                    linewidth=2, label=f'{account} Rating', color='#3498db')
        axes[i].set_title(f'{account} - Rating Evolution Over Time', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('Rating')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

        # Add trend line
        x_numeric = np.arange(len(account_data))
        z = np.polyfit(x_numeric, account_data['cumulative_rating'], 1)
        p = np.poly1d(z)
        axes[i].plot(account_data['datetime'], p(x_numeric),
                    "--", alpha=0.8, color='#e74c3c', label=f'Trend (slope: {z[0]:.1f})')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig('assets/chart2_rating_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 3: Performance by Game Type and Time
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Performance by game type
    game_type_performance = df.groupby(['account', 'game_type'])['result'].apply(lambda x: (x == 'win').mean() * 100).unstack(fill_value=0)
    game_type_performance.plot(kind='bar', ax=ax1, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    ax1.set_title('Win Rate by Game Type', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Account')
    ax1.set_ylabel('Win Rate (%)')
    ax1.legend(title='Game Type')
    ax1.tick_params(axis='x', rotation=45)

    # Performance by color
    color_performance = df.groupby(['account', 'color'])['result'].apply(lambda x: (x == 'win').mean() * 100).unstack(fill_value=0)
    color_performance.plot(kind='bar', ax=ax2, color=['#2c3e50', '#ecf0f1'])
    ax2.set_title('Win Rate by Color', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Account')
    ax2.set_ylabel('Win Rate (%)')
    ax2.legend(title='Color')
    ax2.tick_params(axis='x', rotation=45)

    # Performance by hour of day
    hourly_performance = df.groupby('hour')['result'].apply(lambda x: (x == 'win').mean() * 100)
    hourly_performance.plot(kind='line', ax=ax3, marker='o', color='#9b59b6', linewidth=2)
    ax3.set_title('Win Rate by Hour of Day', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Hour')
    ax3.set_ylabel('Win Rate (%)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(0, 24, 2))

    # Performance by day of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_performance = df.groupby('day_of_week')['result'].apply(lambda x: (x == 'win').mean() * 100).reindex(day_order)
    daily_performance.plot(kind='bar', ax=ax4, color='#e67e22')
    ax4.set_title('Win Rate by Day of Week', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Day of Week')
    ax4.set_ylabel('Win Rate (%)')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('assets/chart3_performance_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 4: Opening Analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Most played openings
    top_openings = df['opening'].value_counts().head(10)
    top_openings.plot(kind='barh', ax=ax1, color='#1abc9c')
    ax1.set_title('Top 10 Most Played Openings', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Games')

    # Best performing openings (min 10 games)
    opening_performance = df.groupby('opening').agg({
        'result': lambda x: (x == 'win').mean() * 100,
        'opening': 'count'
    }).rename(columns={'result': 'win_rate', 'opening': 'game_count'})

    best_openings = opening_performance[opening_performance['game_count'] >= 10].nlargest(10, 'win_rate')
    best_openings['win_rate'].plot(kind='barh', ax=ax2, color='#27ae60')
    ax2.set_title('Best Performing Openings (≥10 games)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Win Rate (%)')

    # ECO code distribution
    eco_counts = df['eco_code'].value_counts().head(15)
    eco_counts.plot(kind='bar', ax=ax3, color='#8e44ad')
    ax3.set_title('Top 15 ECO Codes', fontsize=14, fontweight='bold')
    ax3.set_xlabel('ECO Code')
    ax3.set_ylabel('Number of Games')
    ax3.tick_params(axis='x', rotation=45)

    # Opening performance by account
    for i, account in enumerate(df['account'].unique()):
        account_openings = df[df['account'] == account]['opening'].value_counts().head(5)
        account_openings.plot(kind='barh', ax=ax4, alpha=0.7, label=account)

    ax4.set_title('Top 5 Openings by Account', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Number of Games')
    ax4.legend()

    plt.tight_layout()
    plt.savefig('assets/chart4_opening_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 5: Game Characteristics Analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Game length distribution
    df['total_moves'].hist(bins=30, ax=ax1, alpha=0.7, color='#34495e')
    ax1.axvline(df['total_moves'].mean(), color='red', linestyle='--', label=f'Mean: {df["total_moves"].mean():.1f}')
    ax1.set_title('Game Length Distribution (Total Moves)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Moves')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Termination reasons
    termination_counts = df['termination'].value_counts().head(8)
    termination_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Game Termination Reasons', fontsize=14, fontweight='bold')
    ax2.set_ylabel('')

    # Rating difference vs win rate
    df['elo_diff_bucket'] = pd.cut(df['elo_diff'], bins=[-float('inf'), -200, -100, -50, 0, 50, 100, 200, float('inf')])
    elo_diff_performance = df.groupby('elo_diff_bucket')['result'].apply(lambda x: (x == 'win').mean() * 100)
    elo_diff_performance.plot(kind='bar', ax=ax3, color='#f39c12')
    ax3.set_title('Win Rate vs Rating Difference\n(Negative = Opponent Stronger)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Rating Difference')
    ax3.set_ylabel('Win Rate (%)')
    ax3.tick_params(axis='x', rotation=45)

    # Monthly activity heatmap
    df['year_month'] = df['datetime'].dt.to_period('M')
    monthly_games = df.groupby(['year_month', 'account']).size().unstack(fill_value=0)

    # Create a simple monthly activity chart
    monthly_total = df.groupby('year_month').size()
    monthly_total.plot(kind='line', ax=ax4, marker='o', color='#e74c3c', linewidth=2)
    ax4.set_title('Monthly Game Activity', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Number of Games')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('assets/chart5_game_characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 6: Advanced Insights and Hidden Gems
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Streak analysis
    def calculate_streaks(results):
        streaks = []
        current_streak = 0
        current_type = None

        for result in results:
            if result == current_type:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append((current_type, current_streak))
                current_type = result
                current_streak = 1

        if current_streak > 0:
            streaks.append((current_type, current_streak))

        return streaks

    all_streaks = []
    for account in df['account'].unique():
        account_data = df[df['account'] == account].sort_values('datetime')
        streaks = calculate_streaks(account_data['result'].tolist())
        for streak_type, length in streaks:
            all_streaks.append({'account': account, 'type': streak_type, 'length': length})

    streak_df = pd.DataFrame(all_streaks)
    win_streaks = streak_df[streak_df['type'] == 'win']['length']
    loss_streaks = streak_df[streak_df['type'] == 'loss']['length']

    ax1.hist([win_streaks, loss_streaks], bins=range(1, max(max(win_streaks), max(loss_streaks)) + 2),
             alpha=0.7, label=['Win Streaks', 'Loss Streaks'], color=['#2ecc71', '#e74c3c'])
    ax1.set_title('Win/Loss Streak Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Streak Length')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Performance vs game time
    df['game_hour_category'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24],
                                     labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)'])
    time_performance = df.groupby('game_hour_category')['result'].apply(lambda x: (x == 'win').mean() * 100)
    time_performance.plot(kind='bar', ax=ax2, color='#9b59b6')
    ax2.set_title('Performance by Time of Day Category', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Category')
    ax2.set_ylabel('Win Rate (%)')
    ax2.tick_params(axis='x', rotation=45)

    # Elo progression correlation with account
    for account in df['account'].unique():
        account_data = df[df['account'] == account].sort_values('datetime')
        account_data['game_number'] = range(len(account_data))
        account_data['result_numeric'] = (account_data['result'] == 'win').astype(int)
        account_data['rolling_winrate'] = account_data['result_numeric'].rolling(window=50, min_periods=1).mean() * 100

        ax3.plot(account_data['game_number'], account_data['rolling_winrate'],
                label=f'{account}', linewidth=2, alpha=0.8)

    ax3.set_title('Rolling Win Rate (50-game window)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Game Number')
    ax3.set_ylabel('Win Rate (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Opening diversity over time
    def calculate_opening_diversity(data, window=100):
        diversity_scores = []
        for i in range(window, len(data) + 1):
            recent_openings = data.iloc[i-window:i]['opening'].value_counts()
            diversity = len(recent_openings)  # Number of unique openings
            diversity_scores.append(diversity)
        return diversity_scores

    for account in df['account'].unique():
        account_data = df[df['account'] == account].sort_values('datetime')
        if len(account_data) >= 100:
            diversity = calculate_opening_diversity(account_data)
            game_numbers = range(100, len(account_data) + 1)
            ax4.plot(game_numbers, diversity, label=f'{account}', linewidth=2, alpha=0.8)

    ax4.set_title('Opening Diversity Over Time (100-game window)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Game Number')
    ax4.set_ylabel('Number of Unique Openings')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('assets/chart6_advanced_insights.png', dpi=300, bbox_inches='tight')
    plt.close()

def extract_hidden_gems(df):
    """Extract valuable insights and hidden patterns"""
    insights = []

    # Calculate key statistics
    total_games = len(df)

    for account in df['account'].unique():
        account_data = df[df['account'] == account]
        account_insights = [f"\n=== {account.upper()} ACCOUNT INSIGHTS ==="]

        # Basic stats
        games_count = len(account_data)
        win_rate = (account_data['result'] == 'win').mean() * 100
        current_rating = account_data.sort_values('datetime')['player_elo'].iloc[-1]
        initial_rating = account_data.sort_values('datetime')['player_elo'].iloc[0]
        rating_change = current_rating - initial_rating

        account_insights.extend([
            f"📊 Total Games: {games_count:,}",
            f"🏆 Overall Win Rate: {win_rate:.1f}%",
            f"📈 Rating Journey: {initial_rating} → {current_rating} ({rating_change:+d})",
            ""
        ])

        # Best time to play
        hourly_performance = account_data.groupby('hour')['result'].apply(lambda x: (x == 'win').mean() * 100)
        best_hour = hourly_performance.idxmax()
        best_hour_winrate = hourly_performance.max()
        worst_hour = hourly_performance.idxmin()
        worst_hour_winrate = hourly_performance.min()

        account_insights.extend([
            f"⏰ Best playing time: {best_hour}:00 ({best_hour_winrate:.1f}% win rate)",
            f"⏰ Worst playing time: {worst_hour}:00 ({worst_hour_winrate:.1f}% win rate)",
            ""
        ])

        # Color preference performance
        color_performance = account_data.groupby('color')['result'].apply(lambda x: (x == 'win').mean() * 100)
        if len(color_performance) == 2:
            white_winrate = color_performance.get('white', 0)
            black_winrate = color_performance.get('black', 0)
            color_diff = abs(white_winrate - black_winrate)
            better_color = 'white' if white_winrate > black_winrate else 'black'

            account_insights.extend([
                f"♟️  White pieces win rate: {white_winrate:.1f}%",
                f"♛ Black pieces win rate: {black_winrate:.1f}%",
                f"🎯 Color strength: {color_diff:.1f}% better with {better_color}",
                ""
            ])

        # Game type mastery
        game_type_performance = account_data.groupby('game_type')['result'].apply(lambda x: (x == 'win').mean() * 100)
        best_format = game_type_performance.idxmax()
        best_format_winrate = game_type_performance.max()

        account_insights.extend([
            f"🚀 Strongest format: {best_format.title()} ({best_format_winrate:.1f}% win rate)",
            ""
        ])

        # Opening mastery
        opening_performance = account_data.groupby('opening').agg({
            'result': lambda x: (x == 'win').mean() * 100,
            'opening': 'count'
        }).rename(columns={'result': 'win_rate', 'opening': 'game_count'})

        # Best openings with at least 5 games
        best_openings = opening_performance[opening_performance['game_count'] >= 5].nlargest(3, 'win_rate')
        most_played = opening_performance.nlargest(3, 'game_count')

        account_insights.append("🔥 Top performing openings (≥5 games):")
        for i, (opening, row) in enumerate(best_openings.iterrows(), 1):
            account_insights.append(f"   {i}. {opening}: {row['win_rate']:.1f}% ({int(row['game_count'])} games)")

        account_insights.append("\n📈 Most played openings:")
        for i, (opening, row) in enumerate(most_played.iterrows(), 1):
            account_insights.append(f"   {i}. {opening}: {int(row['game_count'])} games ({row['win_rate']:.1f}% win rate)")

        # Streak analysis
        account_data_sorted = account_data.sort_values('datetime')
        results = account_data_sorted['result'].tolist()

        # Find longest win streak
        max_win_streak = 0
        current_win_streak = 0
        max_loss_streak = 0
        current_loss_streak = 0

        for result in results:
            if result == 'win':
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            elif result == 'loss':
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
            else:  # draw
                current_win_streak = 0
                current_loss_streak = 0

        account_insights.extend([
            "",
            f"🔥 Longest win streak: {max_win_streak} games",
            f"❄️  Longest loss streak: {max_loss_streak} games",
            ""
        ])

        # Performance against stronger/weaker opponents
        stronger_opponents = account_data[account_data['elo_diff'] > 50]
        weaker_opponents = account_data[account_data['elo_diff'] < -50]

        if len(stronger_opponents) > 0:
            strong_opp_winrate = (stronger_opponents['result'] == 'win').mean() * 100
            account_insights.append(f"💪 vs Stronger opponents (+50 elo): {strong_opp_winrate:.1f}% win rate ({len(stronger_opponents)} games)")

        if len(weaker_opponents) > 0:
            weak_opp_winrate = (weaker_opponents['result'] == 'win').mean() * 100
            account_insights.append(f"🎯 vs Weaker opponents (-50 elo): {weak_opp_winrate:.1f}% win rate ({len(weaker_opponents)} games)")

        # Recent form (last 50 games)
        recent_games = account_data_sorted.tail(50)
        recent_winrate = (recent_games['result'] == 'win').mean() * 100
        account_insights.extend([
            "",
            f"📊 Recent form (last 50 games): {recent_winrate:.1f}% win rate"
        ])

        insights.extend(account_insights)
        insights.append("\n" + "="*50 + "\n")

    # Combined insights
    insights.extend([
        "🔍 COMBINED INSIGHTS & HIDDEN GEMS:",
        "",
        f"🎮 Total chess games analyzed: {total_games:,}",
        f"📅 Data spans from {df['date'].min()} to {df['date'].max()}",
    ])

    # Cross-account comparison
    account_comparison = df.groupby('account').agg({
        'result': lambda x: (x == 'win').mean() * 100,
        'player_elo': ['min', 'max', 'mean'],
        'total_moves': 'mean',
        'game_type': lambda x: x.value_counts().index[0]  # Most common game type
    }).round(1)

    insights.append("\n📊 Account Comparison:")
    for account in df['account'].unique():
        account_stats = account_comparison.loc[account]
        insights.append(f"   {account}:")
        insights.append(f"     - Win Rate: {account_stats[('result', '<lambda>')]}%")
        insights.append(f"     - Rating Range: {int(account_stats[('player_elo', 'min')])} - {int(account_stats[('player_elo', 'max')])}")
        insights.append(f"     - Avg Game Length: {account_stats[('total_moves', 'mean')]:.1f} moves")
        insights.append(f"     - Preferred Format: {account_stats[('game_type', '<lambda>')].title()}")

    # Time patterns
    peak_hour = df.groupby('hour').size().idxmax()
    peak_day = df.groupby('day_of_week').size().idxmax()

    insights.extend([
        "",
        f"⏰ Peak playing time: {peak_hour}:00",
        f"📅 Most active day: {peak_day}",
    ])

    # Opening diversity
    total_unique_openings = df['opening'].nunique()
    insights.append(f"🎲 Opening diversity: {total_unique_openings} unique openings played")

    return "\n".join(insights)

def main():
    print("Starting comprehensive chess analysis...")

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

    # Create charts
    print("Creating charts...")
    create_charts(df)

    # Extract insights
    print("Extracting insights...")
    insights = extract_hidden_gems(df)

    # Save insights to file
    with open('chess_insights.txt', 'w', encoding='utf-8') as f:
        f.write(insights)

    print("\nAnalysis complete!")
    print("Generated files:")
    print("- assets/chart1_results_overview.png")
    print("- assets/chart2_rating_evolution.png")
    print("- assets/chart3_performance_patterns.png")
    print("- assets/chart4_opening_analysis.png")
    print("- assets/chart5_game_characteristics.png")
    print("- assets/chart6_advanced_insights.png")
    print("- chess_insights.txt")

    print("\n" + "="*50)
    print("KEY INSIGHTS PREVIEW:")
    print("="*50)
    print(insights[:2000] + "..." if len(insights) > 2000 else insights)

if __name__ == "__main__":
    main()