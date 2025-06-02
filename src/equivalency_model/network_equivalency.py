"""
Network-based league equivalency calculation for English football.
Adapted from NHL analytics methodology.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
import itertools
import logging

logger = logging.getLogger(__name__)


class NetworkEquivalency:
    """Calculate league equivalency factors using network approach."""
    
    def __init__(self, min_transitions: int = 10, confidence_level: float = 0.8):
        """
        Initialize network equivalency calculator.
        
        Args:
            min_transitions: Minimum player transitions required between leagues
            confidence_level: Confidence level for bootstrap intervals
        """
        self.min_transitions = min_transitions
        self.confidence_level = confidence_level
        self.equivalency_factors = {}
        self.network = None
        self.transition_data = None
        
        logger.info(f"NetworkEquivalency initialized (min_transitions={min_transitions})")
    
    def calculate_conversion_factor(self, transitions_df: pd.DataFrame, 
                                  league1: str, league2: str) -> Tuple[Optional[float], int]:
        """
        Calculate conversion factor between two leagues.
        
        Returns:
            Tuple of (conversion_factor, number_of_transitions)
        """
        # Filter for transitions between these specific leagues
        mask = ((transitions_df['league1'] == league1) & 
                (transitions_df['league2'] == league2)) | \
               ((transitions_df['league1'] == league2) & 
                (transitions_df['league2'] == league1))
        
        league_transitions = transitions_df[mask].copy()
        
        if len(league_transitions) < self.min_transitions:
            return None, len(league_transitions)
        
        # Calculate total points and minutes for each league
        total_points1 = 0
        total_minutes1 = 0
        total_points2 = 0
        total_minutes2 = 0
        
        for _, row in league_transitions.iterrows():
            # Determine which stats correspond to which league
            if row['league1'] == league1:
                points1 = row['goals1'] + row['assists1']
                minutes1 = row['minutes1']
                points2 = row['goals2'] + row['assists2']
                minutes2 = row['minutes2']
            else:
                points1 = row['goals2'] + row['assists2']
                minutes1 = row['minutes2']
                points2 = row['goals1'] + row['assists1']
                minutes2 = row['minutes1']
            
            total_points1 += points1
            total_minutes1 += minutes1
            total_points2 += points2
            total_minutes2 += minutes2
        
        # Calculate points per 90 minutes for each league
        if total_minutes1 > 0 and total_minutes2 > 0:
            points_per_90_league1 = (total_points1 / total_minutes1) * 90
            points_per_90_league2 = (total_points2 / total_minutes2) * 90
            
            # Conversion factor: league2_rate / league1_rate
            if points_per_90_league1 > 0:
                conversion_factor = points_per_90_league2 / points_per_90_league1
            else:
                conversion_factor = 0
        else:
            conversion_factor = 0
            
        logger.debug(f"Conversion {league1}->{league2}: {conversion_factor:.3f} ({len(league_transitions)} transitions)")
        return conversion_factor, len(league_transitions)
    
    def calculate_equivalency_factors(self, transitions_df: pd.DataFrame, 
                                    target_league: str = "premier_league") -> Dict[str, Dict]:
        """Calculate equivalency factors for all leagues relative to target."""
        
        # Get all unique leagues
        leagues = set(transitions_df['league1'].unique()) | set(transitions_df['league2'].unique())
        
        # Calculate factors for each league
        factors = {}
        
        # Target league always has factor of 1.0
        factors[target_league] = {
            'factor': 1.0,
            'confidence_interval': (1.0, 1.0),
            'transitions': None
        }
        
        for league in leagues:
            if league == target_league:
                continue
                
            # Calculate direct conversion factor to target league
            factor, n_transitions = self.calculate_conversion_factor(
                transitions_df, league, target_league)
            
            if factor is not None:
                # Simple confidence interval (±10% for now)
                margin = factor * 0.1
                ci = (max(0, factor - margin), factor + margin)
                
                factors[league] = {
                    'factor': factor,
                    'confidence_interval': ci,
                    'transitions': n_transitions
                }
                
                logger.info(f"Equivalency {league}: {factor:.3f} ({n_transitions} transitions)")
            else:
                factors[league] = {
                    'factor': None,
                    'confidence_interval': (None, None),
                    'transitions': n_transitions
                }
                logger.warning(f"Insufficient data for {league}: only {n_transitions} transitions")
        
        self.equivalency_factors = factors
        return factors
    
    def get_equivalency_table(self) -> pd.DataFrame:
        """Get equivalency factors as a formatted table."""
        if not self.equivalency_factors:
            raise ValueError("No equivalency factors calculated. Run calculate_equivalency_factors first.")
        
        table_data = []
        for league, data in self.equivalency_factors.items():
            table_data.append({
                'League': league,
                'Equivalency_Factor': data['factor'],
                'Lower_CI': data['confidence_interval'][0],
                'Upper_CI': data['confidence_interval'][1],
                'Transitions': data['transitions']
            })
        
        df = pd.DataFrame(table_data)
        
        # Sort by equivalency factor (descending)
        df = df.sort_values('Equivalency_Factor', ascending=False, na_last=True)
        
        return df


# Example usage
if __name__ == "__main__":
    eq_model = NetworkEquivalency(min_transitions=5)
    print(f"Initialized NetworkEquivalency with min_transitions={eq_model.min_transitions}")
