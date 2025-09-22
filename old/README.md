# Chess Games Analysis

This repository contains a script for analyzing chess games from a PGN (Portable Game Notation) file. The script extracts various insights and statistics from the games and visualizes them using graphs.

## Features

- Extracts and analyzes game results (wins, losses, draws).
- Tracks rating changes over time for both White and Black players.
- Analyzes the frequency and success rates of different chess openings.
- Examines the distribution of time controls used in the games.
- Provides performance statistics for each color (White and Black).

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ismat-Samadov/chess_games_analyse.git
   cd chess_games_analyse
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Update the PGN file path in the script:**
   In `main.py`, replace `'games.pgn'` with the actual path to your PGN file.

2. **Run the script:**
   ```bash
   python3 main.py
   ```

3. **Output:**
   - The script will print various insights to the console:
     - Overall game results.
     - Total number of games.
     - Performance by color.
     - Top 10 openings and their frequencies.
     - Win rates by opening.
   - The script will generate and display several plots:
     - Bar chart of game results.
     - Line plot of rating changes over time.
     - Bar chart of the top 10 openings.
     - Bar chart of time controls used.
   - A CSV file named `parsed_lichess_games.csv` will be saved in the current directory, containing all parsed game data.

## Example Output

```
Overall Performance:
 Result
1-0        3545
0-1        3224
1/2-1/2     237
Name: count, dtype: int64

Total number of games: 7006

Performance by Color: White Wins: 3545, Black Wins: 3224, Draws: 237

Top 10 Openings:
 Opening
King's Pawn Game: Leonardis Variation             497
Scandinavian Defense: Valencian Variation         318
Scandinavian Defense: Mieses-Kotroc Variation     174
Philidor Defense                                  170
Sicilian Defense                                  151
King's Gambit Declined: Queen's Knight Defense    138
Sicilian Defense: McDonnell Attack                134
King's Gambit                                     122
Modern Defense                                    118
French Defense: Advance Variation                 114
Name: count, dtype: int64

Win Rate by Opening (Top 5):
Wins:
 Opening
King's Pawn Game: Leonardis Variation            258
Scandinavian Defense: Valencian Variation        163
Scandinavian Defense: Mieses-Kotroc Variation     87
Philidor Defense                                  78
King's Gambit                                     73
Name: count, dtype: int64
Losses:
 Opening
King's Pawn Game: Leonardis Variation            223
Scandinavian Defense: Valencian Variation        144
Philidor Defense                                  85
Scandinavian Defense: Mieses-Kotroc Variation     80
Sicilian Defense                                  75
Name: count, dtype: int64
Draws:
 Opening
King's Pawn Game: Leonardis Variation            16
Scandinavian Defense: Valencian Variation        11
Philidor Defense                                  7
Scandinavian Defense: Mieses-Kotroc Variation     7
Queen's Pawn Game: Accelerated London System      5
Name: count, dtype: int64
```

## Contributing

If you have suggestions for improvements or additional features, feel free to create an issue or submit a pull request.