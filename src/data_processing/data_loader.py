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
    
    def _build_file_path(self, data_root: str, league: str, season: str, stat_type: str = "standard") -> Path:
        """Build file path for specific league/season/stat combination."""
        league_name = self.leagues[league]['fbref_name']
        file_name = self.data_files[stat_type]
        
        return Path(data_root) / f"{league_name}_{season}" / file_name
    
    def load_single_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load a single CSV file with error handling."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.debug(f"Loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
    
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
        
        # Remove rows with missing essential data
        essential_cols = ['player', 'games']
        for col in essential_cols:
            if col in df_clean.columns:
                df_clean = df_clean.dropna(subset=[col])
        
        # Filter to common stats available across all leagues
        available_cols = [col for col in self.common_stats if col in df_clean.columns]
        meta_cols = ['league', 'season']
        df_clean = df_clean[available_cols + meta_cols]
        
        logger.debug(f"Cleaned dataframe: {len(df_clean)} rows, {len(df_clean.columns)} columns")
        return df_clean
    
    def identify_transitions(self, df: pd.DataFrame, min_games: int = 5) -> pd.DataFrame:
        """Identify players who played in multiple leagues in the same season."""
        transitions = []
        
        logger.info("Identifying player transitions between leagues...")
        
        for (player, season), group in df.groupby(['player', 'season']):
            # Filter to players with sufficient games in each league
            valid_leagues = group[group['games'] >= min_games]
            
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
                            'games1': stats1['games'],
                            'games2': stats2['games'],
                            'goals1': stats1['goals'],
                            'goals2': stats2['goals'],
                            'assists1': stats1['assists'],
                            'assists2': stats2['assists'],
                            'minutes1': stats1['minutes'],
                            'minutes2': stats2['minutes'],
                            'age': stats1['age']
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
