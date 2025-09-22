import chess.pgn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import re
from collections import Counter, defaultdict
from scipy import stats
from sklearn.cluster import KMeans
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
                total_time = base_time + increment * 40
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

def advanced_time_analysis(df):
    """Advanced time-based analysis with psychological insights"""

    # Convert datetime and add time features
    df['datetime'] = pd.to_datetime(df['utc_date'] + ' ' + df['utc_time'])
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['week_of_year'] = df['datetime'].dt.isocalendar().week
    df['month'] = df['datetime'].dt.month
    df['quarter'] = df['datetime'].dt.quarter
    df['year'] = df['datetime'].dt.year

    # Advanced time categorization
    def get_time_period(hour):
        if 5 <= hour < 9:
            return 'Early Morning'
        elif 9 <= hour < 12:
            return 'Late Morning'
        elif 12 <= hour < 14:
            return 'Lunch Time'
        elif 14 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 20:
            return 'Evening'
        elif 20 <= hour < 23:
            return 'Night'
        else:
            return 'Late Night'

    def get_weekend_weekday(day):
        return 'Weekend' if day in ['Saturday', 'Sunday'] else 'Weekday'

    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'

    df['time_period'] = df['hour'].apply(get_time_period)
    df['weekend_weekday'] = df['day_of_week'].apply(get_weekend_weekday)
    df['season'] = df['month'].apply(get_season)

    # Calculate time since last game
    df_sorted = df.sort_values(['account', 'datetime'])
    df_sorted['time_since_last_game'] = df_sorted.groupby('account')['datetime'].diff()
    df_sorted['time_since_last_game_hours'] = df_sorted['time_since_last_game'].dt.total_seconds() / 3600

    # Session analysis - games played within 1 hour of each other are considered same session
    df_sorted['new_session'] = (df_sorted['time_since_last_game_hours'] > 1) | df_sorted['time_since_last_game_hours'].isna()
    df_sorted['session_id'] = df_sorted.groupby('account')['new_session'].cumsum()

    return df_sorted

def create_advanced_charts(df):
    """Create advanced analysis charts"""

    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

    # Chart 1: Detailed Hourly Performance Heatmap
    plt.figure(figsize=(16, 8))

    # Create performance matrix by account and hour
    performance_matrix = []
    accounts = df['account'].unique()

    for account in accounts:
        account_data = df[df['account'] == account]
        hourly_perf = []
        for hour in range(24):
            hour_data = account_data[account_data['hour'] == hour]
            if len(hour_data) > 0:
                win_rate = (hour_data['result'] == 'win').mean() * 100
                hourly_perf.append(win_rate)
            else:
                hourly_perf.append(np.nan)
        performance_matrix.append(hourly_perf)

    # Create heatmap
    sns.heatmap(performance_matrix,
                xticklabels=[f'{h:02d}:00' for h in range(24)],
                yticklabels=accounts,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn',
                center=50,
                cbar_kws={'label': 'Win Rate (%)'})

    plt.title('Hourly Performance Heatmap by Account', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Account', fontsize=12)
    plt.tight_layout()
    plt.savefig('assets/15_hourly_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 2: Time Period Performance Analysis
    plt.figure(figsize=(14, 8))
    time_period_perf = df.groupby(['account', 'time_period'])['result'].apply(lambda x: (x == 'win').mean() * 100).unstack(fill_value=0)
    time_period_perf.plot(kind='bar', width=0.8)
    plt.title('Performance by Time Period', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Account', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.legend(title='Time Period', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/16_time_period_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 3: Weekend vs Weekday Performance with Volume
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Performance comparison
    weekend_weekday_perf = df.groupby(['account', 'weekend_weekday'])['result'].apply(lambda x: (x == 'win').mean() * 100).unstack()
    weekend_weekday_perf.plot(kind='bar', ax=ax1, color=['#3498db', '#e74c3c'])
    ax1.set_title('Weekend vs Weekday Performance', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Account')
    ax1.set_ylabel('Win Rate (%)')
    ax1.legend(title='Day Type')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Volume comparison
    weekend_weekday_volume = df.groupby(['account', 'weekend_weekday']).size().unstack()
    weekend_weekday_volume.plot(kind='bar', ax=ax2, color=['#9b59b6', '#f39c12'])
    ax2.set_title('Weekend vs Weekday Game Volume', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Account')
    ax2.set_ylabel('Number of Games')
    ax2.legend(title='Day Type')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('assets/17_weekend_weekday_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 4: Seasonal Performance Analysis
    plt.figure(figsize=(12, 8))
    seasonal_data = df.groupby(['account', 'season']).agg({
        'result': lambda x: (x == 'win').mean() * 100,
        'account': 'count'
    }).rename(columns={'result': 'win_rate', 'account': 'game_count'})

    seasonal_pivot = seasonal_data.reset_index().pivot(index='season', columns='account', values='win_rate')
    seasonal_pivot.plot(kind='bar', width=0.7)
    plt.title('Seasonal Performance Analysis', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Season', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.legend(title='Account')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/18_seasonal_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 5: Session Analysis - Performance vs Session Length
    plt.figure(figsize=(14, 8))

    # Calculate session statistics
    session_stats = df.groupby(['account', 'session_id']).agg({
        'result': ['count', lambda x: (x == 'win').mean()],
        'datetime': ['min', 'max']
    }).reset_index()

    session_stats.columns = ['account', 'session_id', 'games_in_session', 'session_win_rate', 'session_start', 'session_end']
    session_stats['session_duration'] = (session_stats['session_end'] - session_stats['session_start']).dt.total_seconds() / 3600

    # Filter sessions with at least 3 games
    session_stats = session_stats[session_stats['games_in_session'] >= 3]

    for account in df['account'].unique():
        account_sessions = session_stats[session_stats['account'] == account]
        if len(account_sessions) > 0:
            plt.scatter(account_sessions['games_in_session'],
                       account_sessions['session_win_rate'] * 100,
                       alpha=0.6, s=60, label=account)

    plt.xlabel('Games in Session', fontsize=12)
    plt.ylabel('Session Win Rate (%)', fontsize=12)
    plt.title('Performance vs Session Length', fontsize=16, fontweight='bold', pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/19_session_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 6: Time Between Games Analysis
    plt.figure(figsize=(14, 8))

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

    gap_performance = time_gap_data.groupby(['account', 'time_gap_category'])['result'].apply(lambda x: (x == 'win').mean() * 100).unstack(fill_value=0)
    gap_performance.plot(kind='bar', width=0.8)
    plt.title('Performance vs Time Since Last Game', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Account', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.legend(title='Time Gap', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/20_time_gap_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 7: Monthly Performance Trends with Confidence Intervals
    plt.figure(figsize=(16, 8))

    for account in df['account'].unique():
        account_data = df[df['account'] == account]
        monthly_stats = account_data.groupby(account_data['datetime'].dt.to_period('M')).agg({
            'result': [lambda x: (x == 'win').mean() * 100, 'count', lambda x: (x == 'win').std() * 100]
        }).reset_index()

        monthly_stats.columns = ['month', 'win_rate', 'game_count', 'win_rate_std']
        monthly_stats = monthly_stats[monthly_stats['game_count'] >= 10]  # At least 10 games per month

        if len(monthly_stats) > 0:
            # Calculate confidence intervals
            monthly_stats['ci'] = 1.96 * monthly_stats['win_rate_std'] / np.sqrt(monthly_stats['game_count'])

            x_pos = range(len(monthly_stats))
            plt.plot(x_pos, monthly_stats['win_rate'], marker='o', linewidth=2, label=account, markersize=6)
            plt.fill_between(x_pos,
                           monthly_stats['win_rate'] - monthly_stats['ci'],
                           monthly_stats['win_rate'] + monthly_stats['ci'],
                           alpha=0.2)

    plt.title('Monthly Performance Trends with 95% Confidence Intervals', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Win Rate (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('assets/21_monthly_trends_ci.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 8: Game Clustering Analysis by Performance Patterns
    plt.figure(figsize=(14, 10))

    # Prepare data for clustering
    clustering_features = []
    clustering_labels = []

    for account in df['account'].unique():
        account_data = df[df['account'] == account]

        # Create feature matrix: hour, day_of_week_num, rating, game_type_encoded
        features = []
        for _, game in account_data.iterrows():
            day_num = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                      'Friday': 4, 'Saturday': 5, 'Sunday': 6}[game['day_of_week']]
            game_type_num = {'bullet': 0, 'blitz': 1, 'rapid': 2, 'classical': 3}[game['game_type']]

            features.append([
                game['hour'] / 24.0,  # Normalize hour
                day_num / 6.0,        # Normalize day
                game['player_elo'] / 2000.0,  # Normalize rating
                game_type_num / 3.0,  # Normalize game type
                1 if game['result'] == 'win' else 0  # Win indicator
            ])
            clustering_labels.append(account)

        clustering_features.extend(features)

    # Perform clustering
    if len(clustering_features) > 100:
        clustering_features = np.array(clustering_features)
        # Use only first 4 features for clustering (exclude result)
        X = clustering_features[:, :4]

        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(X)

        # Create visualization using first two principal components
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('Game Pattern Clustering Analysis\n(Based on Time, Day, Rating, Game Type)',
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)

        # Add cluster centers
        centers_pca = pca.transform(kmeans.cluster_centers_)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('assets/22_game_pattern_clustering.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Chart 9: Performance Volatility Analysis
    plt.figure(figsize=(16, 8))

    for account in df['account'].unique():
        account_data = df[df['account'] == account].sort_values('datetime')

        # Calculate rolling statistics
        account_data['result_numeric'] = (account_data['result'] == 'win').astype(int)
        account_data['rolling_mean'] = account_data['result_numeric'].rolling(window=50, min_periods=10).mean()
        account_data['rolling_std'] = account_data['result_numeric'].rolling(window=50, min_periods=10).std()

        # Plot volatility (standard deviation)
        valid_data = account_data.dropna(subset=['rolling_std'])
        if len(valid_data) > 0:
            plt.plot(range(len(valid_data)), valid_data['rolling_std'],
                    label=f'{account} (50-game rolling std)', linewidth=2, alpha=0.8)

    plt.title('Performance Volatility Over Time (50-game rolling window)',
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Game Number', fontsize=12)
    plt.ylabel('Performance Standard Deviation', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('assets/23_performance_volatility.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Chart 10: Advanced Opening Performance by Time Period
    plt.figure(figsize=(16, 10))

    # Get top 5 openings for each account
    top_openings_by_account = {}
    for account in df['account'].unique():
        account_data = df[df['account'] == account]
        top_openings = account_data['opening'].value_counts().head(5).index.tolist()
        top_openings_by_account[account] = top_openings

    # Combine all top openings
    all_top_openings = []
    for openings in top_openings_by_account.values():
        all_top_openings.extend(openings)
    all_top_openings = list(set(all_top_openings))[:8]  # Limit to 8 for readability

    # Create performance matrix for openings by time period
    opening_time_performance = df[df['opening'].isin(all_top_openings)].groupby(['opening', 'time_period'])['result'].apply(lambda x: (x == 'win').mean() * 100).unstack(fill_value=0)

    # Create heatmap
    sns.heatmap(opening_time_performance,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn',
                center=50,
                cbar_kws={'label': 'Win Rate (%)'})

    plt.title('Opening Performance by Time Period', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Opening', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('assets/24_opening_time_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Starting advanced chess analysis...")

    # Parse both PGN files
    print("Parsing IsmatS.pgn...")
    ismats_games = parse_pgn_file('IsmatS.pgn')

    print("Parsing Cassiny.pgn...")
    cassiny_games = parse_pgn_file('Cassiny.pgn')

    # Combine data
    all_games = ismats_games + cassiny_games
    df = pd.DataFrame(all_games)

    # Advanced time analysis
    print("Performing advanced time analysis...")
    df = advanced_time_analysis(df)

    # Create advanced charts
    print("Creating advanced analysis charts...")
    create_advanced_charts(df)

    print("\nAdvanced analysis complete!")
    print("Generated 10 additional sophisticated charts with deep insights!")

if __name__ == "__main__":
    main()