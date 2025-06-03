"""
Robust equivalency calculation that handles missing data properly.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RobustEquivalency:
    """Calculate equivalency factors with proper data handling."""
    
    def __init__(self, min_transitions: int = 10):
        self.min_transitions = min_transitions
        
    def calculate_league_factor(self, transitions_df: pd.DataFrame, 
                               league1: str, league2: str) -> dict:
        """Calculate conversion factor between two leagues with robust data handling."""
        
        # Filter transitions between these leagues
        mask = ((transitions_df['league1'] == league1) & 
                (transitions_df['league2'] == league2)) | \
               ((transitions_df['league1'] == league2) & 
                (transitions_df['league2'] == league1))
        
        league_transitions = transitions_df[mask].copy()
        
        if len(league_transitions) < self.min_transitions:
            return {
                'factor': None,
                'transitions': len(league_transitions),
                'status': f'Insufficient data: {len(league_transitions)} transitions (need {self.min_transitions})'
            }
        
        # Clean the transition data
        valid_transitions = []
        
        for _, row in league_transitions.iterrows():
            # Determine which league is which
            if row['league1'] == league1:
                goals1, assists1, minutes1 = row['goals1'], row['assists1'], row['minutes1']
                goals2, assists2, minutes2 = row['goals2'], row['assists2'], row['minutes2']
            else:
                goals1, assists1, minutes1 = row['goals2'], row['assists2'], row['minutes2']
                goals2, assists2, minutes2 = row['goals1'], row['assists1'], row['minutes1']
            
            # Only include if we have valid data for both sides
            if (pd.notna(goals1) and pd.notna(goals2) and 
                pd.notna(assists1) and pd.notna(assists2)):
                
                # Use games as proxy for minutes if minutes is missing
                if pd.isna(minutes1) and pd.notna(row.get('games1', 0)):
                    minutes1 = row['games1'] * 60  # Assume 60 min per game average
                if pd.isna(minutes2) and pd.notna(row.get('games2', 0)):
                    minutes2 = row['games2'] * 60
                
                # Only include if we have some measure of playing time
                if pd.notna(minutes1) and pd.notna(minutes2) and minutes1 > 0 and minutes2 > 0:
                    valid_transitions.append({
                        'player': row['player'],
                        'goals1': goals1, 'assists1': assists1, 'minutes1': minutes1,
                        'goals2': goals2, 'assists2': assists2, 'minutes2': minutes2,
                        'points1': goals1 + assists1,
                        'points2': goals2 + assists2
                    })
        
        if len(valid_transitions) < 5:  # Need minimum valid sample
            return {
                'factor': None,
                'transitions': len(league_transitions),
                'valid_transitions': len(valid_transitions),
                'status': f'Insufficient valid data: {len(valid_transitions)} valid of {len(league_transitions)} total'
            }
        
        # Calculate aggregate rates
        total_points1 = sum(t['points1'] for t in valid_transitions)
        total_minutes1 = sum(t['minutes1'] for t in valid_transitions)
        total_points2 = sum(t['points2'] for t in valid_transitions)
        total_minutes2 = sum(t['minutes2'] for t in valid_transitions)
        
        # Points per 90 minutes
        rate1 = (total_points1 / total_minutes1) * 90 if total_minutes1 > 0 else 0
        rate2 = (total_points2 / total_minutes2) * 90 if total_minutes2 > 0 else 0
        
        # Conversion factor: how much more productive is league1 vs league2
        factor = rate1 / rate2 if rate2 > 0 else None
        
        return {
            'factor': factor,
            'transitions': len(league_transitions),
            'valid_transitions': len(valid_transitions),
            'league1_rate': rate1,
            'league2_rate': rate2,
            'status': 'Success',
            'sample_players': [t['player'] for t in valid_transitions[:3]]
        }
    
    def analyze_all_pairs(self, transitions_df: pd.DataFrame) -> dict:
        """Analyze all league pairs."""
        
        results = {}
        
        # Define the pairs we want to analyze
        pairs = [
            ('championship', 'league_one'),
            ('premier_league', 'championship'), 
            ('championship', 'league_two'),
            ('league_one', 'league_two'),
            ('premier_league', 'league_one'),
            ('premier_league', 'league_two')
        ]
        
        for league1, league2 in pairs:
            result = self.calculate_league_factor(transitions_df, league1, league2)
            pair_name = f'{league1}_vs_{league2}'
            results[pair_name] = result
            
        return results

if __name__ == "__main__":
    pass
