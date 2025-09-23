#!/usr/bin/env python3
"""
Comprehensive Chess Analysis Script
Generates 18 high-quality analytical charts from combined PGN data
"""

import chess.pgn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import re
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for high-quality charts
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ChessAnalyzer:
    def __init__(self, pgn_file):
        self.pgn_file = pgn_file
        self.games_data = []
        self.players = set()

    def parse_pgn(self):
        """Parse PGN file and extract game data"""
        print("Parsing PGN file...")

        with open(self.pgn_file, 'r', encoding='utf-8') as f:
            game_count = 0
            while True:
                try:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break

                    headers = game.headers

                    # Extract basic game info
                    white_player = headers.get('White', 'Unknown')
                    black_player = headers.get('Black', 'Unknown')
                    result = headers.get('Result', '*')
                    date = headers.get('Date', '????')
                    time_control = headers.get('TimeControl', 'Unknown')
                    termination = headers.get('Termination', 'Unknown')

                    # Extract ratings
                    white_elo = int(headers.get('WhiteElo', 0)) if headers.get('WhiteElo', '0').isdigit() else 0
                    black_elo = int(headers.get('BlackElo', 0)) if headers.get('BlackElo', '0').isdigit() else 0

                    # Extract opening
                    opening = headers.get('Opening', 'Unknown')
                    eco = headers.get('ECO', 'Unknown')

                    # Parse date
                    try:
                        if date != '????':
                            game_date = datetime.strptime(date.replace('?', '01'), '%Y.%m.%d')
                        else:
                            game_date = None
                    except:
                        game_date = None

                    # Count moves
                    moves = len(list(game.mainline_moves()))

                    # Extract time information if available
                    utc_date = headers.get('UTCDate')
                    utc_time = headers.get('UTCTime')
                    game_time = None
                    if utc_date and utc_time:
                        try:
                            game_time = datetime.strptime(f"{utc_date} {utc_time}", '%Y.%m.%d %H:%M:%S')
                        except:
                            pass

                    self.players.add(white_player)
                    self.players.add(black_player)

                    game_data = {
                        'white_player': white_player,
                        'black_player': black_player,
                        'result': result,
                        'white_elo': white_elo,
                        'black_elo': black_elo,
                        'date': game_date,
                        'time': game_time,
                        'time_control': time_control,
                        'termination': termination,
                        'opening': opening,
                        'eco': eco,
                        'moves': moves,
                        'game_number': game_count
                    }

                    self.games_data.append(game_data)
                    game_count += 1

                    if game_count % 1000 == 0:
                        print(f"Processed {game_count} games...")

                except Exception as e:
                    print(f"Error processing game {game_count}: {e}")
                    continue

        print(f"Total games processed: {len(self.games_data)}")
        self.df = pd.DataFrame(self.games_data)
        return self.df

    def prepare_player_data(self):
        """Prepare data focusing on our main players"""
        main_players = ['IsmatS', 'Cassiny']

        # Create separate rows for each player perspective
        player_games = []

        for _, game in self.df.iterrows():
            # Add white perspective
            if game['white_player'] in main_players:
                player_game = game.copy()
                player_game['player'] = game['white_player']
                player_game['player_color'] = 'white'
                player_game['player_elo'] = game['white_elo']
                player_game['opponent'] = game['black_player']
                player_game['opponent_elo'] = game['black_elo']

                if game['result'] == '1-0':
                    player_game['player_result'] = 'win'
                elif game['result'] == '0-1':
                    player_game['player_result'] = 'loss'
                else:
                    player_game['player_result'] = 'draw'

                player_games.append(player_game)

            # Add black perspective
            if game['black_player'] in main_players:
                player_game = game.copy()
                player_game['player'] = game['black_player']
                player_game['player_color'] = 'black'
                player_game['player_elo'] = game['black_elo']
                player_game['opponent'] = game['white_player']
                player_game['opponent_elo'] = game['white_elo']

                if game['result'] == '0-1':
                    player_game['player_result'] = 'win'
                elif game['result'] == '1-0':
                    player_game['player_result'] = 'loss'
                else:
                    player_game['player_result'] = 'draw'

                player_games.append(player_game)

        self.player_df = pd.DataFrame(player_games)

        # Add derived columns
        if len(self.player_df) > 0:
            self.player_df['rating_diff'] = self.player_df['player_elo'] - self.player_df['opponent_elo']
            self.player_df['win_rate'] = (self.player_df['player_result'] == 'win').astype(int)

            # Add time-based features
            if self.player_df['date'].notna().any():
                self.player_df['month'] = self.player_df['date'].dt.month
                self.player_df['weekday'] = self.player_df['date'].dt.dayofweek
                self.player_df['is_weekend'] = self.player_df['weekday'].isin([5, 6])

            if self.player_df['time'].notna().any():
                self.player_df['hour'] = self.player_df['time'].dt.hour

        return self.player_df

    def create_chart_1_win_rates(self):
        """Chart 1: Overall Win Rates by Player"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Win rate by player
        win_rates = self.player_df.groupby('player')['player_result'].value_counts(normalize=True).unstack(fill_value=0)
        win_rates.plot(kind='bar', ax=ax1, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
        ax1.set_title('Win Rate Distribution by Player', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Player')
        ax1.set_ylabel('Percentage')
        ax1.legend(['Draw', 'Loss', 'Win'])
        ax1.tick_params(axis='x', rotation=0)

        # Overall statistics
        stats = self.player_df.groupby('player').agg({
            'player_result': lambda x: (x == 'win').mean(),
            'player_elo': 'mean',
            'game_number': 'count'
        }).round(3)

        # Table
        ax2.axis('tight')
        ax2.axis('off')
        table_data = []
        for player in stats.index:
            table_data.append([
                player,
                f"{stats.loc[player, 'player_result']:.1%}",
                f"{stats.loc[player, 'player_elo']:.0f}",
                f"{stats.loc[player, 'game_number']:.0f}"
            ])

        table = ax2.table(cellText=table_data,
                         colLabels=['Player', 'Win Rate', 'Avg Rating', 'Games'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        ax2.set_title('Player Statistics', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig('assets/01_win_rates_overview.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_2_rating_evolution(self):
        """Chart 2: Rating Evolution Over Time"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Filter data with valid dates and ratings
        valid_data = self.player_df[
            (self.player_df['date'].notna()) &
            (self.player_df['player_elo'] > 0)
        ].copy()

        if len(valid_data) == 0:
            plt.text(0.5, 0.5, 'No valid rating data available',
                    ha='center', va='center', transform=fig.transFigure, fontsize=16)
            plt.savefig('assets/02_rating_evolution.png', dpi=300, bbox_inches='tight')
            plt.close()
            return

        valid_data = valid_data.sort_values('date')

        # Plot for each player
        for i, player in enumerate(['IsmatS', 'Cassiny']):
            player_data = valid_data[valid_data['player'] == player]
            if len(player_data) > 0:
                # Calculate rolling average
                if len(player_data) > 10:
                    player_data['rating_ma'] = player_data['player_elo'].rolling(window=10, min_periods=1).mean()
                else:
                    player_data['rating_ma'] = player_data['player_elo']

                axes[i].plot(player_data['date'], player_data['player_elo'],
                           alpha=0.3, color='lightblue', linewidth=0.5)
                axes[i].plot(player_data['date'], player_data['rating_ma'],
                           color='darkblue', linewidth=2, label=f'{player} Rating (MA)')

                axes[i].set_title(f'{player} Rating Evolution', fontsize=14, fontweight='bold')
                axes[i].set_xlabel('Date')
                axes[i].set_ylabel('Rating')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()

        plt.tight_layout()
        plt.savefig('assets/02_rating_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_3_opening_analysis(self):
        """Chart 3: Opening Analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Top openings by frequency
        opening_counts = self.player_df['opening'].value_counts().head(15)
        opening_counts.plot(kind='barh', ax=ax1, color='skyblue')
        ax1.set_title('Top 15 Most Played Openings', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Games')

        # Opening success rates (minimum 5 games)
        opening_performance = self.player_df.groupby('opening').agg({
            'win_rate': 'mean',
            'game_number': 'count'
        })
        opening_performance = opening_performance[opening_performance['game_number'] >= 5]
        top_openings = opening_performance.nlargest(10, 'win_rate')

        top_openings['win_rate'].plot(kind='bar', ax=ax2, color='lightgreen')
        ax2.set_title('Best Performing Openings (Min 5 Games)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Win Rate')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('assets/03_opening_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_4_time_control_performance(self):
        """Chart 4: Performance by Time Control"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Time control distribution
        tc_counts = self.player_df['time_control'].value_counts().head(10)
        tc_counts.plot(kind='pie', ax=ax1, autopct='%1.1f%%')
        ax1.set_title('Games by Time Control', fontsize=14, fontweight='bold')
        ax1.set_ylabel('')

        # Performance by time control
        tc_performance = self.player_df.groupby('time_control').agg({
            'win_rate': 'mean',
            'game_number': 'count'
        })
        tc_performance = tc_performance[tc_performance['game_number'] >= 10]

        if len(tc_performance) > 0:
            tc_performance['win_rate'].plot(kind='bar', ax=ax2, color='orange')
            ax2.set_title('Win Rate by Time Control (Min 10 Games)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Win Rate')
            ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('assets/04_time_control_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_5_color_performance(self):
        """Chart 5: Performance by Color (White vs Black)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Overall color performance
        color_perf = self.player_df.groupby(['player', 'player_color'])['win_rate'].mean().unstack()
        color_perf.plot(kind='bar', ax=ax1, color=['black', 'white'], edgecolor='gray')
        ax1.set_title('Win Rate by Color', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Win Rate')
        ax1.legend(['Black', 'White'])
        ax1.tick_params(axis='x', rotation=0)

        # Color preference analysis
        color_dist = self.player_df.groupby('player')['player_color'].value_counts(normalize=True).unstack()
        color_dist.plot(kind='bar', ax=ax2, color=['black', 'white'], edgecolor='gray')
        ax2.set_title('Color Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Percentage of Games')
        ax2.legend(['Black', 'White'])
        ax2.tick_params(axis='x', rotation=0)

        plt.tight_layout()
        plt.savefig('assets/05_color_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_6_termination_analysis(self):
        """Chart 6: Game Termination Analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Termination reasons distribution
        term_counts = self.player_df['termination'].value_counts()
        term_counts.plot(kind='pie', ax=ax1, autopct='%1.1f%%')
        ax1.set_title('Game Termination Reasons', fontsize=14, fontweight='bold')
        ax1.set_ylabel('')

        # Win rate by termination type
        term_winrate = self.player_df.groupby('termination')['win_rate'].mean()
        term_winrate.plot(kind='bar', ax=ax2, color='coral')
        ax2.set_title('Win Rate by Termination Type', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Win Rate')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('assets/06_termination_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_7_time_of_day_performance(self):
        """Chart 7: Performance by Time of Day"""
        if 'hour' not in self.player_df.columns or self.player_df['hour'].isna().all():
            # Create placeholder chart
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No time data available', ha='center', va='center', fontsize=16)
            ax.set_title('Performance by Hour of Day', fontsize=14, fontweight='bold')
            plt.savefig('assets/07_time_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Games by hour
        hour_counts = self.player_df['hour'].value_counts().sort_index()
        hour_counts.plot(kind='bar', ax=ax1, color='lightblue')
        ax1.set_title('Games Played by Hour', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Number of Games')

        # Performance by hour
        hour_performance = self.player_df.groupby('hour')['win_rate'].mean()
        hour_performance.plot(kind='line', marker='o', ax=ax2, color='red', linewidth=2, markersize=6)
        ax2.set_title('Win Rate by Hour of Day', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Win Rate')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('assets/07_time_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_8_monthly_trends(self):
        """Chart 8: Monthly Activity and Performance Trends"""
        if 'month' not in self.player_df.columns or self.player_df['month'].isna().all():
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No date data available', ha='center', va='center', fontsize=16)
            ax.set_title('Monthly Performance Trends', fontsize=14, fontweight='bold')
            plt.savefig('assets/08_monthly_trends.png', dpi=300, bbox_inches='tight')
            plt.close()
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Monthly game count
        monthly_games = self.player_df.groupby(['player', 'month']).size().unstack(fill_value=0)
        monthly_games.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Games Played by Month', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Player')
        ax1.set_ylabel('Number of Games')
        ax1.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=0)

        # Monthly win rates
        monthly_winrate = self.player_df.groupby(['player', 'month'])['win_rate'].mean().unstack()
        monthly_winrate.plot(kind='line', marker='o', ax=ax2, linewidth=2, markersize=6)
        ax2.set_title('Win Rate by Month', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Win Rate')
        ax2.legend(title='Player')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('assets/08_monthly_trends.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_9_opponent_strength(self):
        """Chart 9: Performance vs Opponent Strength"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Opponent rating distribution
        self.player_df['opponent_elo'].hist(bins=30, ax=ax1, color='lightcoral', alpha=0.7)
        ax1.set_title('Opponent Rating Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Opponent Rating')
        ax1.set_ylabel('Frequency')

        # Win rate vs rating difference
        self.player_df['rating_diff_bin'] = pd.cut(self.player_df['rating_diff'],
                                                  bins=[-np.inf, -200, -100, -50, 0, 50, 100, 200, np.inf],
                                                  labels=['<-200', '-200:-100', '-100:-50', '-50:0',
                                                         '0:50', '50:100', '100:200', '>200'])

        rating_diff_perf = self.player_df.groupby('rating_diff_bin')['win_rate'].mean()
        rating_diff_perf.plot(kind='bar', ax=ax2, color='green')
        ax2.set_title('Win Rate vs Rating Difference', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Rating Difference (Player - Opponent)')
        ax2.set_ylabel('Win Rate')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('assets/09_opponent_strength.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_10_game_length(self):
        """Chart 10: Game Length Analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Game length distribution
        self.player_df['moves'].hist(bins=30, ax=ax1, color='purple', alpha=0.7)
        ax1.set_title('Game Length Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Moves')
        ax1.set_ylabel('Frequency')

        # Win rate by game length
        self.player_df['moves_bin'] = pd.cut(self.player_df['moves'],
                                            bins=[0, 20, 40, 60, 80, 100, np.inf],
                                            labels=['0-20', '21-40', '41-60', '61-80', '81-100', '100+'])

        moves_perf = self.player_df.groupby('moves_bin')['win_rate'].mean()
        moves_perf.plot(kind='bar', ax=ax2, color='orange')
        ax2.set_title('Win Rate by Game Length', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Moves')
        ax2.set_ylabel('Win Rate')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('assets/10_game_length.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_11_performance_streaks(self):
        """Chart 11: Performance Streaks Analysis"""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        for i, player in enumerate(['IsmatS', 'Cassiny']):
            player_data = self.player_df[self.player_df['player'] == player].copy()
            if len(player_data) == 0:
                continue

            player_data = player_data.sort_values('game_number')

            # Calculate rolling win rate
            window_size = min(20, len(player_data) // 4)
            if window_size < 5:
                window_size = 5

            player_data['rolling_winrate'] = player_data['win_rate'].rolling(
                window=window_size, min_periods=1).mean()

            # Plot rolling performance
            axes[i].plot(range(len(player_data)), player_data['rolling_winrate'],
                        linewidth=2, label=f'{player} Rolling Win Rate')
            axes[i].fill_between(range(len(player_data)), player_data['rolling_winrate'],
                               alpha=0.3)
            axes[i].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Line')
            axes[i].set_title(f'{player} Performance Trend', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Game Number')
            axes[i].set_ylabel('Rolling Win Rate')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('assets/11_performance_streaks.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_12_weekday_performance(self):
        """Chart 12: Weekday vs Weekend Performance"""
        if 'weekday' not in self.player_df.columns or self.player_df['weekday'].isna().all():
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No weekday data available', ha='center', va='center', fontsize=16)
            ax.set_title('Weekday vs Weekend Performance', fontsize=14, fontweight='bold')
            plt.savefig('assets/12_weekday_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Performance by day of week
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_perf = self.player_df.groupby('weekday')['win_rate'].mean()
        weekday_perf.index = [weekday_names[i] for i in weekday_perf.index]

        weekday_perf.plot(kind='bar', ax=ax1, color='lightblue')
        ax1.set_title('Win Rate by Day of Week', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Win Rate')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)

        # Weekend vs Weekday
        weekend_perf = self.player_df.groupby(['player', 'is_weekend'])['win_rate'].mean().unstack()
        weekend_perf.columns = ['Weekday', 'Weekend']
        weekend_perf.plot(kind='bar', ax=ax2, color=['lightcoral', 'lightgreen'])
        ax2.set_title('Weekend vs Weekday Performance', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Win Rate')
        ax2.tick_params(axis='x', rotation=0)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('assets/12_weekday_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_13_seasonal_performance(self):
        """Chart 13: Seasonal Performance Analysis"""
        if 'month' not in self.player_df.columns or self.player_df['month'].isna().all():
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No seasonal data available', ha='center', va='center', fontsize=16)
            ax.set_title('Seasonal Performance', fontsize=14, fontweight='bold')
            plt.savefig('assets/13_seasonal_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            return

        # Define seasons
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                     9: 'Fall', 10: 'Fall', 11: 'Fall'}

        self.player_df['season'] = self.player_df['month'].map(season_map)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Games by season
        season_counts = self.player_df['season'].value_counts()
        season_counts.plot(kind='pie', ax=ax1, autopct='%1.1f%%')
        ax1.set_title('Games by Season', fontsize=14, fontweight='bold')
        ax1.set_ylabel('')

        # Performance by season
        season_perf = self.player_df.groupby(['player', 'season'])['win_rate'].mean().unstack()
        season_perf.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Win Rate by Season', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Win Rate')
        ax2.tick_params(axis='x', rotation=0)
        ax2.legend(title='Season')

        plt.tight_layout()
        plt.savefig('assets/13_seasonal_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_14_opening_heatmap(self):
        """Chart 14: Opening Success Rate Heatmap"""
        # Get top openings with sufficient games
        opening_stats = self.player_df.groupby(['player', 'opening']).agg({
            'win_rate': 'mean',
            'game_number': 'count'
        }).reset_index()

        # Filter openings with at least 3 games
        opening_stats = opening_stats[opening_stats['game_number'] >= 3]

        if len(opening_stats) == 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'Insufficient opening data for heatmap', ha='center', va='center', fontsize=16)
            ax.set_title('Opening Performance Heatmap', fontsize=14, fontweight='bold')
            plt.savefig('assets/14_opening_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            return

        # Get top openings for each player
        top_openings = []
        for player in opening_stats['player'].unique():
            player_openings = opening_stats[opening_stats['player'] == player].nlargest(10, 'game_number')
            top_openings.extend(player_openings['opening'].tolist())

        top_openings = list(set(top_openings))[:15]  # Limit to 15 openings

        # Create pivot table
        heatmap_data = opening_stats[opening_stats['opening'].isin(top_openings)]
        pivot_table = heatmap_data.pivot(index='opening', columns='player', values='win_rate')

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', center=0.5,
                   fmt='.2f', ax=ax, cbar_kws={'label': 'Win Rate'})
        ax.set_title('Opening Performance Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Player')
        ax.set_ylabel('Opening')

        plt.tight_layout()
        plt.savefig('assets/14_opening_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_15_performance_volatility(self):
        """Chart 15: Performance Volatility Analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Calculate performance volatility for each player
        volatility_data = []
        for player in self.player_df['player'].unique():
            player_data = self.player_df[self.player_df['player'] == player].sort_values('game_number')
            if len(player_data) < 10:
                continue

            # Calculate rolling standard deviation of win rate
            window_size = min(20, len(player_data) // 4)
            rolling_std = player_data['win_rate'].rolling(window=window_size, min_periods=5).std()

            volatility_data.append({
                'player': player,
                'avg_volatility': rolling_std.mean(),
                'max_volatility': rolling_std.max(),
                'games': len(player_data)
            })

        if volatility_data:
            vol_df = pd.DataFrame(volatility_data)
            vol_df.set_index('player')['avg_volatility'].plot(kind='bar', ax=ax1, color='red')
            ax1.set_title('Average Performance Volatility', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Standard Deviation')
            ax1.tick_params(axis='x', rotation=0)

            # Performance consistency over time
            for player in self.player_df['player'].unique():
                player_data = self.player_df[self.player_df['player'] == player].sort_values('game_number')
                if len(player_data) < 10:
                    continue

                window_size = min(20, len(player_data) // 4)
                rolling_winrate = player_data['win_rate'].rolling(window=window_size, min_periods=1).mean()

                ax2.plot(range(len(rolling_winrate)), rolling_winrate, label=player, linewidth=2)

            ax2.set_title('Performance Consistency Over Time', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Game Number')
            ax2.set_ylabel('Rolling Win Rate')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('assets/15_performance_volatility.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_16_time_investment(self):
        """Chart 16: Time Investment Analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Estimated time per game based on time control
        time_mapping = {
            '180+0': 3, '300+0': 5, '600+0': 10, '900+0': 15, '1800+0': 30,
            '180+2': 3, '300+3': 5, '600+5': 10, '900+10': 15
        }

        # Extract estimated time from time control
        def estimate_time(tc):
            if pd.isna(tc) or tc == 'Unknown':
                return 5  # Default estimate
            for pattern, time in time_mapping.items():
                if pattern in str(tc):
                    return time
            return 5

        self.player_df['estimated_time'] = self.player_df['time_control'].apply(estimate_time)

        # Time investment by player
        time_investment = self.player_df.groupby('player').agg({
            'estimated_time': 'sum',
            'game_number': 'count'
        })
        time_investment['avg_time_per_game'] = time_investment['estimated_time'] / time_investment['game_number']

        time_investment['estimated_time'].plot(kind='bar', ax=ax1, color='lightblue')
        ax1.set_title('Total Estimated Time Investment (minutes)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Minutes')
        ax1.tick_params(axis='x', rotation=0)

        time_investment['avg_time_per_game'].plot(kind='bar', ax=ax2, color='orange')
        ax2.set_title('Average Time per Game (minutes)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Minutes')
        ax2.tick_params(axis='x', rotation=0)

        plt.tight_layout()
        plt.savefig('assets/16_time_investment.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_17_comprehensive_dashboard(self):
        """Chart 17: Comprehensive Performance Dashboard"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 1. Overall win rates
        ax1 = fig.add_subplot(gs[0, 0])
        win_rates = self.player_df.groupby('player')['win_rate'].mean()
        win_rates.plot(kind='bar', ax=ax1, color=['#1f77b4', '#ff7f0e'])
        ax1.set_title('Win Rates', fontweight='bold')
        ax1.tick_params(axis='x', rotation=0)

        # 2. Game count
        ax2 = fig.add_subplot(gs[0, 1])
        game_counts = self.player_df.groupby('player').size()
        game_counts.plot(kind='bar', ax=ax2, color=['#2ca02c', '#d62728'])
        ax2.set_title('Total Games', fontweight='bold')
        ax2.tick_params(axis='x', rotation=0)

        # 3. Average rating
        ax3 = fig.add_subplot(gs[0, 2])
        avg_ratings = self.player_df.groupby('player')['player_elo'].mean()
        avg_ratings.plot(kind='bar', ax=ax3, color=['#9467bd', '#8c564b'])
        ax3.set_title('Average Rating', fontweight='bold')
        ax3.tick_params(axis='x', rotation=0)

        # 4. Color preference
        ax4 = fig.add_subplot(gs[0, 3])
        color_perf = self.player_df.groupby(['player', 'player_color'])['win_rate'].mean().unstack()
        if color_perf is not None and not color_perf.empty:
            color_perf.plot(kind='bar', ax=ax4, color=['black', 'white'], edgecolor='gray')
            ax4.set_title('Win Rate by Color', fontweight='bold')
            ax4.tick_params(axis='x', rotation=0)
            ax4.legend(['Black', 'White'])

        # 5. Most played openings
        ax5 = fig.add_subplot(gs[1, :2])
        top_openings = self.player_df['opening'].value_counts().head(8)
        top_openings.plot(kind='barh', ax=ax5, color='skyblue')
        ax5.set_title('Most Played Openings', fontweight='bold')

        # 6. Performance by rating difference
        ax6 = fig.add_subplot(gs[1, 2:])
        if 'rating_diff_bin' in self.player_df.columns:
            rating_diff_perf = self.player_df.groupby('rating_diff_bin')['win_rate'].mean()
            rating_diff_perf.plot(kind='bar', ax=ax6, color='green')
            ax6.set_title('Win Rate vs Rating Difference', fontweight='bold')
            ax6.tick_params(axis='x', rotation=45)
            ax6.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)

        # 7. Termination reasons
        ax7 = fig.add_subplot(gs[2, :2])
        term_counts = self.player_df['termination'].value_counts().head(6)
        term_counts.plot(kind='pie', ax=ax7, autopct='%1.1f%%')
        ax7.set_title('Game Termination Reasons', fontweight='bold')
        ax7.set_ylabel('')

        # 8. Performance trend
        ax8 = fig.add_subplot(gs[2, 2:])
        for player in self.player_df['player'].unique():
            player_data = self.player_df[self.player_df['player'] == player].sort_values('game_number')
            if len(player_data) > 10:
                window_size = min(20, len(player_data) // 4)
                rolling_winrate = player_data['win_rate'].rolling(window=window_size, min_periods=1).mean()
                ax8.plot(range(len(rolling_winrate)), rolling_winrate, label=player, linewidth=2)

        ax8.set_title('Performance Trend', fontweight='bold')
        ax8.set_xlabel('Game Number')
        ax8.set_ylabel('Rolling Win Rate')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        plt.suptitle('Chess Performance Dashboard', fontsize=20, fontweight='bold', y=0.98)
        plt.savefig('assets/17_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chart_18_advanced_metrics(self):
        """Chart 18: Advanced Performance Metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Performance vs expected (based on rating difference)
        if 'rating_diff' in self.player_df.columns:
            # Calculate expected score using Elo formula
            self.player_df['expected_score'] = 1 / (1 + 10 ** (-self.player_df['rating_diff'] / 400))
            self.player_df['performance_vs_expected'] = self.player_df['win_rate'] - self.player_df['expected_score']

            perf_vs_exp = self.player_df.groupby('player')['performance_vs_expected'].mean()
            perf_vs_exp.plot(kind='bar', ax=ax1, color=['green' if x > 0 else 'red' for x in perf_vs_exp.values])
            ax1.set_title('Performance vs Expected', fontweight='bold')
            ax1.set_ylabel('Actual - Expected')
            ax1.tick_params(axis='x', rotation=0)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # 2. Win rate by game length categories
        if 'moves_bin' in self.player_df.columns:
            length_perf = self.player_df.groupby('moves_bin')['win_rate'].mean()
            length_perf.plot(kind='line', marker='o', ax=ax2, color='purple', linewidth=2, markersize=6)
            ax2.set_title('Win Rate by Game Length', fontweight='bold')
            ax2.set_ylabel('Win Rate')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)

        # 3. Consistency metric (std dev of results)
        consistency = self.player_df.groupby('player')['win_rate'].std()
        consistency.plot(kind='bar', ax=ax3, color='orange')
        ax3.set_title('Performance Consistency (Lower = More Consistent)', fontweight='bold')
        ax3.set_ylabel('Standard Deviation')
        ax3.tick_params(axis='x', rotation=0)

        # 4. Score distribution
        score_dist = self.player_df.groupby(['player', 'player_result']).size().unstack(fill_value=0)
        score_dist_pct = score_dist.div(score_dist.sum(axis=1), axis=0)
        score_dist_pct.plot(kind='bar', stacked=True, ax=ax4,
                           color=['#ff6b6b', '#feca57', '#48dbfb'])
        ax4.set_title('Result Distribution', fontweight='bold')
        ax4.set_ylabel('Percentage')
        ax4.tick_params(axis='x', rotation=0)
        ax4.legend(['Draw', 'Loss', 'Win'])

        plt.tight_layout()
        plt.savefig('assets/18_advanced_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_all_charts(self):
        """Generate all 18 charts"""
        print("Generating comprehensive chess analysis charts...")

        charts = [
            (self.create_chart_1_win_rates, "Win Rates Overview"),
            (self.create_chart_2_rating_evolution, "Rating Evolution"),
            (self.create_chart_3_opening_analysis, "Opening Analysis"),
            (self.create_chart_4_time_control_performance, "Time Control Performance"),
            (self.create_chart_5_color_performance, "Color Performance"),
            (self.create_chart_6_termination_analysis, "Termination Analysis"),
            (self.create_chart_7_time_of_day_performance, "Time of Day Performance"),
            (self.create_chart_8_monthly_trends, "Monthly Trends"),
            (self.create_chart_9_opponent_strength, "Opponent Strength Analysis"),
            (self.create_chart_10_game_length, "Game Length Analysis"),
            (self.create_chart_11_performance_streaks, "Performance Streaks"),
            (self.create_chart_12_weekday_performance, "Weekday Performance"),
            (self.create_chart_13_seasonal_performance, "Seasonal Performance"),
            (self.create_chart_14_opening_heatmap, "Opening Performance Heatmap"),
            (self.create_chart_15_performance_volatility, "Performance Volatility"),
            (self.create_chart_16_time_investment, "Time Investment Analysis"),
            (self.create_chart_17_comprehensive_dashboard, "Comprehensive Dashboard"),
            (self.create_chart_18_advanced_metrics, "Advanced Metrics")
        ]

        for i, (chart_func, name) in enumerate(charts, 1):
            try:
                print(f"Creating Chart {i:2d}: {name}")
                chart_func()
            except Exception as e:
                print(f"Error creating Chart {i:2d} ({name}): {e}")
                continue

        print("All charts generated successfully!")

def main():
    analyzer = ChessAnalyzer('combined_games.pgn')

    # Parse the PGN file
    df = analyzer.parse_pgn()

    # Prepare player-focused data
    player_df = analyzer.prepare_player_data()

    print(f"Total games: {len(df)}")
    print(f"Player games: {len(player_df)}")
    print(f"Players: {analyzer.players}")

    # Generate all charts
    analyzer.generate_all_charts()

if __name__ == "__main__":
    main()