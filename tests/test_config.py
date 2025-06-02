"""Tests for configuration loading."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import load_config, load_model_params, get_leagues

def test_load_config():
    """Test configuration loading."""
    config = load_config()
    
    # Check required sections exist
    assert "leagues" in config
    assert "common_stats" in config
    assert "position_groups" in config
    assert "seasons" in config
    
    # Check league structure
    assert "premier_league" in config["leagues"]
    assert config["leagues"]["premier_league"]["tier"] == 1
    assert config["leagues"]["premier_league"]["equivalency_factor"] == 1.0

def test_load_model_params():
    """Test model parameters loading."""
    params = load_model_params()
    
    # Check required sections
    assert "equivalency_model" in params
    assert "prospect_model" in params
    assert "success_definitions" in params
    
    # Check specific parameters
    assert params["equivalency_model"]["min_transitions"] == 10
    assert params["prospect_model"]["cv_folds"] == 5

def test_get_leagues():
    """Test league getter function."""
    leagues = get_leagues()
    
    assert len(leagues) == 4
    assert "premier_league" in leagues
    assert "championship" in leagues
    assert "league_one" in leagues
    assert "league_two" in leagues

if __name__ == "__main__":
    # Run tests
    test_load_config()
    test_load_model_params() 
    test_get_leagues()
    print("All configuration tests passed!")
