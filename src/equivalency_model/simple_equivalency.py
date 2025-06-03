"""
Simple direct equivalency calculation for English football leagues.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SimpleEquivalency:
    """Calculate direct equivalency factors between league pairs."""
    
    def __init__(self, min_transitions: int = 10):
        self.min_transitions = min_transitions
        
    def calculate_direct_factor(self, transitions_df: pd.DataFrame, 
                               league1: str, league2: str) -> dict:
        """Calculate direct conversion factor between two specific leagues."""
        
        # Filter transitions between these two leagues
        mask = ((transitions_df['league1'] == league1) & 
                (transitions_df['league2'] == league2)) | \
               ((transitions_df['league1'] == league2) & 
                (transitions_df['league2'] == league1))
        
        league_transitions = transitions_df[mask].copy()
        
        if len(league_transitions) < self.min_transitions:
            return {
                'factor': None,
                'transitions': len(league_transitions),
                'reason': f'Only {len(league_transitions)} transitions (need {self.min_transitions})'
            }
        
        # Calculate total performance metrics for each league
        total_points1 = 0
        total_minutes1 = 0
        total_points2 = 0
        total_minutes2 = 0
        
        for _, row in league_transitions.iterrows():
            if row['league1'] == league1:
                points1 = row['goals1'] + row['assists1']
                minutes1 = max(row['minutes1'], 1)  # Avoid division by zero
                points2 = row['goals2'] + row['assists2']
                minutes2 = max(row['minutes2'], 1)
            else:
                points1 = row['goals2'] + row['assists2']
                minutes1 = max(row['minutes2'], 1)
                points2 = row['goals1'] + row['assists1']
                minutes2 = max(row['minutes1'], 1)
            
            total_points1 += points1
            total_minutes1 += minutes1
            total_points2 += points2
            total_minutes2 += minutes2
        
        # Calculate points per 90 minutes
        points_per_90_league1 = (total_points1 / total_minutes1) * 90
        points_per_90_league2 = (total_points2 / total_minutes2) * 90
        
        # Conversion factor: how much is league1 worth relative to league2
        if points_per_90_league2 > 0:
            factor = points_per_90_league1 / points_per_90_league2
        else:
            factor = 0
        
        return {
            'factor': factor,
            'transitions': len(league_transitions),
            'league1_rate': points_per_90_league1,
            'league2_rate': points_per_90_league2,
            'sample_players': league_transitions['player'].head(3).tolist()
        }
    
    def analyze_all_transitions(self, transitions_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze all possible league pair transitions."""
        
        # Get unique leagues
        leagues = set(transitions_df['league1'].unique()) | set(transitions_df['league2'].unique())
        leagues = sorted(list(leagues))
        
        results = []
        
        for i, league1 in enumerate(leagues):
            for league2 in leagues[i+1:]:
                result = self.calculate_direct_factor(transitions_df, league1, league2)
                
                results.append({
                    'League_Pair': f'{league1} ↔ {league2}',
                    'League1': league1,
                    'League2': league2,
                    'Conversion_Factor': result['factor'],
                    'Transitions': result['transitions'],
                    'Status': 'Calculated' if result['factor'] is not None else 'Insufficient Data'
                })
        
        return pd.DataFrame(results)

def main():
    """Run the simple equivalency analysis."""
    
    # This will be called from the main script
    pass

if __name__ == "__main__":
    main()
