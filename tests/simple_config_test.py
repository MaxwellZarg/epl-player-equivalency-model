"""Simple configuration test without pytest."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_config_loading():
    """Test basic configuration loading."""
    try:
        from utils.config import load_config, load_model_params
        
        print("Testing configuration loading...")
        
        # Test league config
        config = load_config()
        print(f"[OK] Loaded league config with {len(config['leagues'])} leagues")
        
        # Test model params
        params = load_model_params()
        print(f"[OK] Loaded model params with {len(params)} sections")
        
        # Test specific values
        assert config['leagues']['premier_league']['tier'] == 1
        assert params['equivalency_model']['min_transitions'] == 10
        
        print("[OK] All configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Configuration test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_config_loading()
    if success:
        print("\n[SUCCESS] Your repository is properly configured!")
    else:
        print("\n[WARNING] There might be configuration issues to fix")
