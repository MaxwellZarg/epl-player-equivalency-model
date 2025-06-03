"""
Data loading and preprocessing utilities for EPL analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, List, Tuple, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess football data from multiple leagues."""
    
    def __init__(self, config_path: str = "config/league_mappings.yaml"):
        """Initialize with configuration file."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        self.common_stats = self.config['common_stats']
        self.leagues = self.config['leagues']
        self.data_files = self.config['data_files']
        self.seasons = self.config['seasons']
        
        logger.info(f"DataLoader initialized with {len(self.leagues)} leagues and {len(self.seasons)} seasons")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_best_stat_file_for_league(self, league: str) -> str:
        """Get the best available stat file for a league."""
        league_name = self.leagues[league]['fbref_name']
        
        # EPL and Championship have standard stats
        if league in ['premier_league', 'championship']:
            return 'standard'
        # League One and Two use shooting stats (has goals)
        else:
            return 'shooting'
    
    def load_and_combine_stats(self, data_root: str, league: str, season: str) -> pd.DataFrame:
        """Load and combine multiple stat files to get all needed data."""
        league_name = self.leagues[league]['fbref_name']
        season_dir = Path(data_root) / f"{league_name}_{season}"
        
        if not season_dir.exists():
            raise FileNotFoundError(f"Season directory not found: {season_dir}")
        
        # Start with the main stats file
        main_stat_type = self.get_best_stat_file_for_league(league)
        main_file = season_dir / self.data_files[main_stat_type]
        
        if not main_file.exists():
            raise FileNotFoundError(f"Main stats file not found: {main_file}")
        
        # Load main dataframe
        df_main = pd.read_csv(main_file)
        logger.debug(f"Loaded main stats from {main_file}: {len(df_main)} players")
        
        # For leagues without standard stats, we need to get assists from passing stats
        if league in ['league_one', 'league_two']:
            try:
                passing_file = season_dir / self.data_files['passing']
                if passing_file.exists():
                    df_passing = pd.read_csv(passing_file)
                    # Merge to get assists
                    if 'assists' in df_passing.columns and 'player' in df_passing.columns:
                        df_main = df_main.merge(
                            df_passing[['player', 'assists']], 
                            on='player', 
                            how='left'
                        )
                        logger.debug(f"Added assists from passing stats")
            except Exception as e:
                logger.warning(f"Could not load passing stats for {league} {season}: {e}")
        
        # Get minutes from playingtime stats if not available
        if 'minutes' not in df_main.columns:
            try:
                playtime_file = season_dir / self.data_files['playingtime']
                if playtime_file.exists():
                    df_playtime = pd.read_csv(playtime_file)
                    if 'minutes' in df_playtime.columns and 'player' in df_playtime.columns:
                        df_main = df_main.merge(
                            df_playtime[['player', 'minutes', 'games']], 
                            on='player', 
                            how='left'
                        )
                        logger.debug(f"Added minutes/games from playingtime stats")
            except Exception as e:
                logger.warning(f"Could not load playingtime stats for {league} {season}: {e}")
        
        return df_main
    
    def clean_dataframe(self, df: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
        """Clean and standardize dataframe."""
        df_clean = df.copy()
        
        # Remove duplicate columns
        df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]
        
        # Add league and season information
        df_clean['league'] = league
        df_clean['season'] = season
        
        # Clean player names
        if 'player' in df_clean.columns:
            df_clean['player'] = df_clean['player'].str.strip()
            # Remove any player entries that are just headers or empty
            df_clean = df_clean[df_clean['player'].notna()]
            df_clean = df_clean[df_clean['player'] != 'Player']
        
        # Convert numeric columns
        numeric_cols = ['age', 'games', 'minutes', 'goals', 'assists', 'cards_yellow', 'cards_red']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Fill missing assists with 0 for leagues that don't track them separately
        if 'assists' not in df_clean.columns:
            df_clean['assists'] = 0
        
        # Remove rows with missing essential data
        essential_cols = ['player']
        for col in essential_cols:
            if col in df_clean.columns:
                df_clean = df_clean.dropna(subset=[col])
        
        # Filter to available common stats
        available_cols = [col for col in self.common_stats if col in df_clean.columns]
        meta_cols = ['league', 'season']
        
        # Always include these critical columns even if not in common_stats
        critical_cols = ['goals', 'assists', 'minutes', 'games']
        for col in critical_cols:
            if col in df_clean.columns and col not in available_cols:
                available_cols.append(col)
        
        df_clean = df_clean[available_cols + meta_cols]
        
        logger.debug(f"Cleaned dataframe: {len(df_clean)} rows, {len(df_clean.columns)} columns")
        return df_clean
    
    def load_league_season(self, data_root: str, league: str, season: str) -> pd.DataFrame:
        """Load data for a specific league and season."""
        try:
            df = self.load_and_combine_stats(data_root, league, season)
            df_clean = self.clean_dataframe(df, league, season)
            
            logger.info(f"Loaded {league} {season}: {len(df_clean)} players")
            return df_clean
            
        except FileNotFoundError as e:
            logger.warning(f"Data not found for {league} {season}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading {league} {season}: {e}")
            return pd.DataFrame()
    
    def load_all_data(self, data_root: str, leagues: Optional[List[str]] = None, 
                     seasons: Optional[List[str]] = None) -> pd.DataFrame:
        """Load data for all specified leagues and seasons."""
        
        # Use all leagues/seasons if not specified
        leagues = leagues or list(self.leagues.keys())
        seasons = seasons or self.seasons
        
        all_data = []
        total_files = len(leagues) * len(seasons)
        loaded_files = 0
        
        logger.info(f"Loading data for {len(leagues)} leagues, {len(seasons)} seasons ({total_files} combinations)")
        
        for league in leagues:
            for season in seasons:
                df = self.load_league_season(data_root, league, season)
                if not df.empty:
                    all_data.append(df)
                    loaded_files += 1
        
        if not all_data:
            raise ValueError("No data could be loaded")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Successfully loaded {loaded_files}/{total_files} combinations: {len(combined_df)} total players")
        
        return combined_df
    
    def identify_transitions(self, df: pd.DataFrame, min_games: int = 3) -> pd.DataFrame:
        """Identify players who played in multiple leagues in the same season."""
        transitions = []
        
        logger.info("Identifying player transitions between leagues...")
        
        for (player, season), group in df.groupby(['player', 'season']):
            # Filter to players with sufficient games in each league (or sufficient minutes)
            valid_leagues = group[
                (group.get('games', 0) >= min_games) | 
                (group.get('minutes', 0) >= min_games * 30)  # Assume 30 min per game minimum
            ]
            
            if len(valid_leagues) > 1:  # Player in multiple leagues
                leagues = valid_leagues['league'].unique()
                
                # Create all pairwise combinations
                for i, league1 in enumerate(leagues):
                    for league2 in leagues[i+1:]:
                        # Get stats for each league
                        stats1 = valid_leagues[valid_leagues['league'] == league1].iloc[0]
                        stats2 = valid_leagues[valid_leagues['league'] == league2].iloc[0]
                        
                        transition = {
                            'player': player,
                            'season': season,
                            'league1': league1,
                            'league2': league2,
                            'games1': stats1.get('games', 0),
                            'games2': stats2.get('games', 0),
                            'goals1': stats1.get('goals', 0),
                            'goals2': stats2.get('goals', 0),
                            'assists1': stats1.get('assists', 0),
                            'assists2': stats2.get('assists', 0),
                            'minutes1': stats1.get('minutes', 0),
                            'minutes2': stats2.get('minutes', 0),
                            'age': stats1.get('age', 0)
                        }
                        transitions.append(transition)
        
        transitions_df = pd.DataFrame(transitions)
        logger.info(f"Found {len(transitions_df)} player transitions")
        
        return transitions_df


# Example usage
if __name__ == "__main__":
    loader = DataLoader()
    print(f"Initialized DataLoader with {len(loader.leagues)} leagues")
    print(f"Available seasons: {loader.seasons}")
    print(f"Common stats: {loader.common_stats}")
