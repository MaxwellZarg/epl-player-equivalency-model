"""Configuration loading utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_file: str = "config/league_mappings.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def load_model_params(config_file: str = "config/model_parameters.yaml") -> Dict[str, Any]:
    """Load model parameters from YAML file."""
    return load_config(config_file)

def get_leagues() -> Dict[str, Any]:
    """Get league configuration."""
    config = load_config()
    return config['leagues']

def get_common_stats() -> list:
    """Get list of common statistics across leagues."""
    config = load_config()
    return config['common_stats']

def get_position_groups() -> Dict[str, list]:
    """Get position groupings."""
    config = load_config()
    return config['position_groups']
