"""
Initial Data Exploration Notebook

This script will become a Jupyter notebook for exploring the EPL data
when it becomes available.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path("..").parent / "src"))

from data_processing.data_loader import DataLoader
from equivalency_model.network_equivalency import NetworkEquivalency

def explore_data():
    """Explore the loaded data structure."""
    
    print("=== EPL Data Exploration ===")
    print()
    
    # Initialize data loader
    loader = DataLoader()
    
    print(f"Configured leagues: {list(loader.leagues.keys())}")
    print(f"Available seasons: {loader.seasons}")
    print(f"Common statistics: {loader.common_stats}")
    print()
    
    # When data is available, uncomment and modify this section:
    """
    # Load sample data
    try:
        df = loader.load_all_data(
            data_root="../../data/raw",
            leagues=['premier_league', 'championship'],
            seasons=['2023-2024']
        )
        
        print(f"Loaded {len(df)} player records")
        print(f"Leagues in data: {df['league'].unique()}")
        print(f"Seasons in data: {df['season'].unique()}")
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(df.describe())
        
        # League summary
        summary = loader.get_league_summary(df)
        print("\nLeague Summary:")
        print(summary)
        
        # Identify transitions
        transitions = loader.identify_transitions(df)
        print(f"\nFound {len(transitions)} player transitions")
        
        if len(transitions) > 0:
            print("\nTransition Summary:")
            print(transitions.groupby(['league1', 'league2']).size())
            
            # Calculate equivalency factors
            eq_model = NetworkEquivalency(min_transitions=3)
            factors = eq_model.calculate_equivalency_factors(transitions)
            
            print("\nEquivalency Factors:")
            table = eq_model.get_equivalency_table()
            print(table)
            
    except FileNotFoundError:
        print("Data files not found. This is expected until data is uploaded.")
    except Exception as e:
        print(f"Error loading data: {e}")
    """
    
    print("Ready for data analysis once files are available!")

if __name__ == "__main__":
    explore_data()
