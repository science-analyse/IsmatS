import chess.pgn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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

def generate_advanced_insights(df):
    """Generate comprehensive advanced insights"""

    # Convert datetime and add time features
    df['datetime'] = pd.to_datetime(df['utc_date'] + ' ' + df['utc_time'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year

    # Advanced time categorization
    def get_time_period(hour):
        if 5 <= hour < 9: return 'Early Morning'
        elif 9 <= hour < 12: return 'Late Morning'
        elif 12 <= hour < 14: return 'Lunch Time'
        elif 14 <= hour < 17: return 'Afternoon'
        elif 17 <= hour < 20: return 'Evening'
        elif 20 <= hour < 23: return 'Night'
        else: return 'Late Night'

    def get_season(month):
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        else: return 'Autumn'

    df['time_period'] = df['hour'].apply(get_time_period)
    df['season'] = df['month'].apply(get_season)
    df['weekend_weekday'] = df['day_of_week'].apply(lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday')

    # Calculate time since last game
    df_sorted = df.sort_values(['account', 'datetime'])
    df_sorted['time_since_last_game'] = df_sorted.groupby('account')['datetime'].diff()
    df_sorted['time_since_last_game_hours'] = df_sorted['time_since_last_game'].dt.total_seconds() / 3600

    insights = []
    insights.append("🔬 ADVANCED CHESS ANALYSIS - DEEP INSIGHTS")
    insights.append("=" * 60)
    insights.append("")

    # Detailed hourly analysis
    insights.append("⏰ CIRCADIAN RHYTHM ANALYSIS")
    insights.append("-" * 30)

    for account in df['account'].unique():
        account_data = df[df['account'] == account]
        hourly_performance = account_data.groupby('hour').agg({
            'result': [lambda x: (x == 'win').mean() * 100, 'count']
        }).round(2)
        hourly_performance.columns = ['win_rate', 'game_count']

        # Find peak and worst hours with sufficient data
        valid_hours = hourly_performance[hourly_performance['game_count'] >= 10]

        if len(valid_hours) > 0:
            best_hour = valid_hours['win_rate'].idxmax()
            worst_hour = valid_hours['win_rate'].idxmin()
            best_winrate = valid_hours.loc[best_hour, 'win_rate']
            worst_winrate = valid_hours.loc[worst_hour, 'win_rate']

            insights.append(f"\n🎯 {account} Circadian Performance:")
            insights.append(f"   Peak Hour: {best_hour}:00 ({best_winrate:.1f}% win rate)")
            insights.append(f"   Worst Hour: {worst_hour}:00 ({worst_winrate:.1f}% win rate)")
            insights.append(f"   Performance Swing: {best_winrate - worst_winrate:.1f}%")

            # Morning vs Evening person analysis
            morning_hours = [6, 7, 8, 9, 10, 11]
            evening_hours = [18, 19, 20, 21, 22, 23]

            morning_data = account_data[account_data['hour'].isin(morning_hours)]
            evening_data = account_data[account_data['hour'].isin(evening_hours)]

            if len(morning_data) >= 20 and len(evening_data) >= 20:
                morning_wr = (morning_data['result'] == 'win').mean() * 100
                evening_wr = (evening_data['result'] == 'win').mean() * 100

                tendency = "Morning Person" if morning_wr > evening_wr else "Evening Person"
                diff = abs(morning_wr - evening_wr)

                insights.append(f"   Chronotype: {tendency} ({diff:.1f}% difference)")

    # Time period analysis
    insights.append("\n\n🌅 TIME PERIOD MASTERY")
    insights.append("-" * 25)

    for account in df['account'].unique():
        account_data = df[df['account'] == account]
        period_performance = account_data.groupby('time_period').agg({
            'result': [lambda x: (x == 'win').mean() * 100, 'count']
        }).round(2)
        period_performance.columns = ['win_rate', 'game_count']

        # Filter periods with enough games
        valid_periods = period_performance[period_performance['game_count'] >= 15]

        if len(valid_periods) > 0:
            best_period = valid_periods['win_rate'].idxmax()
            worst_period = valid_periods['win_rate'].idxmin()

            insights.append(f"\n💪 {account} Time Period Dominance:")
            insights.append(f"   Strongest: {best_period} ({valid_periods.loc[best_period, 'win_rate']:.1f}%)")
            insights.append(f"   Weakest: {worst_period} ({valid_periods.loc[worst_period, 'win_rate']:.1f}%)")

    # Weekend vs Weekday psychology
    insights.append("\n\n📅 WEEKEND VS WEEKDAY PSYCHOLOGY")
    insights.append("-" * 35)

    for account in df['account'].unique():
        account_data = df[df['account'] == account]
        weekend_weekday_stats = account_data.groupby('weekend_weekday').agg({
            'result': [lambda x: (x == 'win').mean() * 100, 'count'],
            'total_moves': 'mean',
            'player_elo': 'mean'
        }).round(2)

        weekend_weekday_stats.columns = ['win_rate', 'game_count', 'avg_moves', 'avg_rating']

        if 'Weekend' in weekend_weekday_stats.index and 'Weekday' in weekend_weekday_stats.index:
            weekend_wr = weekend_weekday_stats.loc['Weekend', 'win_rate']
            weekday_wr = weekend_weekday_stats.loc['Weekday', 'win_rate']
            weekend_games = weekend_weekday_stats.loc['Weekend', 'game_count']
            weekday_games = weekend_weekday_stats.loc['Weekday', 'game_count']

            insights.append(f"\n🎮 {account} Weekend vs Weekday:")
            insights.append(f"   Weekend: {weekend_wr:.1f}% WR ({weekend_games} games)")
            insights.append(f"   Weekday: {weekday_wr:.1f}% WR ({weekday_games} games)")

            preference = "Weekend Warrior" if weekend_wr > weekday_wr else "Weekday Grinder"
            insights.append(f"   Profile: {preference}")

    # Session analysis
    insights.append("\n\n🎯 SESSION PERFORMANCE PATTERNS")
    insights.append("-" * 32)

    df_sorted['new_session'] = (df_sorted['time_since_last_game_hours'] > 2) | df_sorted['time_since_last_game_hours'].isna()
    df_sorted['session_id'] = df_sorted.groupby('account')['new_session'].cumsum()

    for account in df['account'].unique():
        account_data = df_sorted[df_sorted['account'] == account]

        # Session statistics
        session_stats = account_data.groupby('session_id').agg({
            'result': ['count', lambda x: (x == 'win').mean()],
            'datetime': ['min', 'max']
        }).reset_index()

        session_stats.columns = ['session_id', 'games_in_session', 'session_wr', 'session_start', 'session_end']
        session_stats['session_duration'] = (session_stats['session_end'] - session_stats['session_start']).dt.total_seconds() / 3600

        # Filter meaningful sessions
        meaningful_sessions = session_stats[session_stats['games_in_session'] >= 3]

        if len(meaningful_sessions) > 10:
            avg_session_length = meaningful_sessions['games_in_session'].mean()
            avg_session_wr = meaningful_sessions['session_wr'].mean() * 100

            # Performance by session length
            short_sessions = meaningful_sessions[meaningful_sessions['games_in_session'] <= 5]
            long_sessions = meaningful_sessions[meaningful_sessions['games_in_session'] >= 10]

            insights.append(f"\n📊 {account} Session Analysis:")
            insights.append(f"   Avg Session Length: {avg_session_length:.1f} games")
            insights.append(f"   Avg Session WR: {avg_session_wr:.1f}%")

            if len(short_sessions) >= 5 and len(long_sessions) >= 5:
                short_wr = short_sessions['session_wr'].mean() * 100
                long_wr = long_sessions['session_wr'].mean() * 100

                session_type = "Sprint Player" if short_wr > long_wr else "Marathon Player"
                insights.append(f"   Playing Style: {session_type}")

    # Seasonal patterns
    insights.append("\n\n🌍 SEASONAL PERFORMANCE PATTERNS")
    insights.append("-" * 32)

    for account in df['account'].unique():
        account_data = df[df['account'] == account]
        seasonal_stats = account_data.groupby('season').agg({
            'result': [lambda x: (x == 'win').mean() * 100, 'count']
        }).round(2)
        seasonal_stats.columns = ['win_rate', 'game_count']

        # Filter seasons with enough data
        valid_seasons = seasonal_stats[seasonal_stats['game_count'] >= 50]

        if len(valid_seasons) >= 2:
            best_season = valid_seasons['win_rate'].idxmax()
            worst_season = valid_seasons['win_rate'].idxmin()

            insights.append(f"\n🌱 {account} Seasonal Performance:")
            insights.append(f"   Best Season: {best_season} ({valid_seasons.loc[best_season, 'win_rate']:.1f}%)")
            insights.append(f"   Worst Season: {worst_season} ({valid_seasons.loc[worst_season, 'win_rate']:.1f}%)")

    # Advanced opening insights
    insights.append("\n\n🧠 ADVANCED OPENING PSYCHOLOGY")
    insights.append("-" * 32)

    for account in df['account'].unique():
        account_data = df[df['account'] == account]

        # Opening diversity by time period
        opening_diversity = account_data.groupby('time_period')['opening'].nunique()
        most_diverse_period = opening_diversity.idxmax()

        # Opening performance by color
        white_openings = account_data[account_data['color'] == 'white']
        black_openings = account_data[account_data['color'] == 'black']

        if len(white_openings) >= 50 and len(black_openings) >= 50:
            white_diversity = white_openings['opening'].nunique()
            black_diversity = black_openings['opening'].nunique()

            insights.append(f"\n♟️  {account} Opening Psychology:")
            insights.append(f"   Most Creative Period: {most_diverse_period}")
            insights.append(f"   White Repertoire: {white_diversity} openings")
            insights.append(f"   Black Repertoire: {black_diversity} openings")

            repertoire_type = "Specialist" if (white_diversity + black_diversity) < 100 else "Generalist"
            insights.append(f"   Style: {repertoire_type}")

    # Performance consistency analysis
    insights.append("\n\n📈 PERFORMANCE CONSISTENCY ANALYSIS")
    insights.append("-" * 35)

    for account in df['account'].unique():
        account_data = df[df['account'] == account].sort_values('datetime')

        # Calculate rolling statistics
        if len(account_data) >= 100:
            account_data['result_numeric'] = (account_data['result'] == 'win').astype(int)
            account_data['rolling_wr'] = account_data['result_numeric'].rolling(window=50, min_periods=25).mean() * 100
            account_data['rolling_std'] = account_data['result_numeric'].rolling(window=50, min_periods=25).std() * 100

            avg_volatility = account_data['rolling_std'].mean()

            consistency_level = "Very Consistent" if avg_volatility < 45 else "Moderately Consistent" if avg_volatility < 50 else "High Volatility"

            insights.append(f"\n📊 {account} Consistency Profile:")
            insights.append(f"   Performance Volatility: {avg_volatility:.1f}%")
            insights.append(f"   Consistency Level: {consistency_level}")

    # Hidden gems and actionable insights
    insights.append("\n\n💎 HIDDEN GEMS & ACTIONABLE INSIGHTS")
    insights.append("-" * 40)

    for account in df['account'].unique():
        account_data = df[df['account'] == account]

        insights.append(f"\n🎯 {account} Strategic Recommendations:")

        # Best day/time combination
        day_hour_performance = account_data.groupby(['day_of_week', 'hour']).agg({
            'result': [lambda x: (x == 'win').mean() * 100, 'count']
        }).reset_index()
        day_hour_performance.columns = ['day', 'hour', 'win_rate', 'game_count']

        # Filter combinations with at least 5 games
        valid_combinations = day_hour_performance[day_hour_performance['game_count'] >= 5]

        if len(valid_combinations) > 0:
            best_combo = valid_combinations.loc[valid_combinations['win_rate'].idxmax()]
            insights.append(f"   🌟 Golden Time: {best_combo['day']} at {int(best_combo['hour'])}:00 ({best_combo['win_rate']:.1f}% WR)")

        # Opening specialization opportunity
        opening_stats = account_data.groupby('opening').agg({
            'result': [lambda x: (x == 'win').mean() * 100, 'count']
        }).reset_index()
        opening_stats.columns = ['opening', 'win_rate', 'game_count']

        # Find high-potential openings (good WR, few games)
        potential_openings = opening_stats[
            (opening_stats['win_rate'] >= 60) &
            (opening_stats['game_count'] >= 3) &
            (opening_stats['game_count'] <= 10)
        ].sort_values('win_rate', ascending=False)

        if len(potential_openings) > 0:
            top_potential = potential_openings.iloc[0]
            insights.append(f"   🚀 Opening to Develop: {top_potential['opening']} ({top_potential['win_rate']:.1f}% WR in {int(top_potential['game_count'])} games)")

        # Time management insight
        if 'time_since_last_game_hours' in account_data.columns:
            rest_performance = account_data[
                (account_data['time_since_last_game_hours'] >= 1) &
                (account_data['time_since_last_game_hours'] <= 168)
            ].copy()

            if len(rest_performance) >= 50:
                def get_rest_category(hours):
                    if hours < 6: return 'Quick Break'
                    elif hours < 24: return 'Day Break'
                    else: return 'Long Break'

                rest_performance['rest_category'] = rest_performance['time_since_last_game_hours'].apply(get_rest_category)
                rest_analysis = rest_performance.groupby('rest_category')['result'].apply(lambda x: (x == 'win').mean() * 100)

                optimal_rest = rest_analysis.idxmax()
                insights.append(f"   ⏱️  Optimal Rest Period: {optimal_rest} ({rest_analysis.max():.1f}% WR)")

    return "\n".join(insights)

def main():
    print("Generating advanced insights...")

    # Parse both PGN files
    ismats_games = parse_pgn_file('IsmatS.pgn')
    cassiny_games = parse_pgn_file('Cassiny.pgn')

    # Combine data
    all_games = ismats_games + cassiny_games
    df = pd.DataFrame(all_games)

    # Generate insights
    insights = generate_advanced_insights(df)

    # Save to file
    with open('advanced_chess_insights.txt', 'w', encoding='utf-8') as f:
        f.write(insights)

    print("Advanced insights generated and saved to 'advanced_chess_insights.txt'")

    # Preview
    print("\n" + "="*60)
    print("PREVIEW OF ADVANCED INSIGHTS:")
    print("="*60)
    print(insights[:3000] + "..." if len(insights) > 3000 else insights)

if __name__ == "__main__":
    main()