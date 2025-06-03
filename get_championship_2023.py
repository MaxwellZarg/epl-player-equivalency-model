"""
Focused scraper to get the missing Championship 2023-2024 standard stats
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import logging
from datetime import datetime
import random
from typing import Dict, Optional
import re

class ChampionshipScraper:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Try different possible Chrome driver paths
        possible_paths = [
            '/opt/homebrew/bin/chromedriver',  # Mac Homebrew
            '/usr/local/bin/chromedriver',     # Linux/Mac
            'C:/chromedriver/chromedriver.exe', # Windows
            'chromedriver.exe',                # Current directory
            'chromedriver'                     # PATH
        ]
        
        self.chrome_driver_path = None
        for path in possible_paths:
            if os.path.exists(path):
                self.chrome_driver_path = path
                break
        
        if not self.chrome_driver_path:
            print("Chrome driver not found. Please install chromedriver or update the path.")
            self.chrome_driver_path = 'chromedriver'  # Hope it's in PATH
        
        self.data_dir = os.path.join(self.base_dir, 'data', 'raw')
        self.setup_logging()
        
        # Standard stats configuration
        self.standard_stats = {
            'url_part': 'stats',
            'possible_table_ids': ['stats_standard', 'stats_standard_9', 'stats_standard_10'],
            'expected_columns': [
                'ranker', 'player', 'nationality', 'position', 'team', 'age', 'birth_year',
                'games', 'games_starts', 'minutes', 'minutes_90s', 'goals', 'assists',
                'goals_assists', 'goals_pens', 'pens_made', 'pens_att', 'cards_yellow',
                'cards_red', 'xg', 'npxg', 'xg_assist', 'npxg_xg_assist', 'progressive_carries',
                'progressive_passes', 'progressive_passes_received', 'goals_per90', 'assists_per90',
                'goals_assists_per90', 'goals_pens_per90', 'goals_assists_pens_per90',
                'xg_per90', 'xg_assist_per90', 'xg_xg_assist_per90', 'npxg_per90',
                'npxg_xg_assist_per90'
            ]
        }

    def setup_logging(self):
        log_dir = os.path.join(self.base_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'championship_scraper_{datetime.now().strftime("%Y%m%d")}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ChampionshipScraper')

    def get_driver(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        try:
            service = Service(self.chrome_driver_path)
            driver = webdriver.Chrome(service=service, options=chrome_options)
        except Exception as e:
            self.logger.error(f"Failed to start Chrome with service: {e}")
            # Try without service (if chromedriver is in PATH)
            driver = webdriver.Chrome(options=chrome_options)
        
        driver.set_page_load_timeout(30)
        return driver

    def find_table(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Find the standard stats table"""
        # Try all possible table IDs
        for table_id in self.standard_stats['possible_table_ids']:
            table = soup.find('table', {'id': table_id})
            if table:
                self.logger.info(f"Found table with ID: {table_id}")
                return table
        
        # Fallback: find any table with player column
        tables = soup.find_all('table')
        for table in tables:
            if table.find('th', {'data-stat': 'player'}):
                self.logger.info("Found table using fallback method")
                return table
        
        return None

    def wait_for_table(self, driver: webdriver.Chrome, timeout: int = 15) -> bool:
        """Wait for table to load"""
        try:
            for table_id in self.standard_stats['possible_table_ids']:
                try:
                    WebDriverWait(driver, timeout).until(
                        EC.presence_of_element_located((By.ID, table_id))
                    )
                    self.logger.info(f"Table {table_id} loaded successfully")
                    return True
                except TimeoutException:
                    continue
            
            # Fallback: wait for any table with player data
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "th[data-stat='player']"))
            )
            self.logger.info("Table loaded using fallback method")
            return True
            
        except TimeoutException:
            self.logger.error("Timeout waiting for table to load")
            return False

    def clean_value(self, value: str) -> str:
        """Clean and standardize data values"""
        if not value or value == '-':
            return ''
        
        value = value.strip()
        
        # Remove percentage signs but keep the number
        value = re.sub(r'%$', '', value)
        
        # Handle ratio format (e.g., "5-2")
        if '-' in value and len(value.split('-')) == 2:
            try:
                parts = value.split('-')
                if len(parts) == 2 and all(p.replace('.', '').isdigit() for p in parts):
                    num, den = map(float, parts)
                    return str(num/den if den != 0 else 0)
            except:
                pass
        
        return value

    def extract_player_data(self, row: BeautifulSoup) -> Optional[Dict]:
        """Extract player data from a table row"""
        try:
            stats = {}
            
            # Get player ID if available
            player_td = row.find('td', {'data-stat': 'player'})
            if player_td and 'data-append-csv' in player_td.attrs:
                stats['player_id'] = player_td['data-append-csv']
            
            # Extract all expected columns
            for col in self.standard_stats['expected_columns']:
                element = row.find(['th', 'td'], {'data-stat': col})
                if element:
                    if col == 'player' and element.find('a'):
                        stats[col] = element.find('a').text.strip()
                    elif col == 'nationality':
                        value = element.text.strip()
                        # Take first 3 characters for nationality code
                        stats[col] = value[:3] if value else ''
                    else:
                        stats[col] = self.clean_value(element.text)
                else:
                    stats[col] = ''
            
            # Only return if we have a player name
            if stats.get('player'):
                return stats
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting player data: {str(e)}")
            return None

    def scrape_championship_2023_24(self, retry_count: int = 3) -> pd.DataFrame:
        """Scrape Championship 2023-24 standard stats"""
        
        # Championship league ID is 10
        url = 'https://fbref.com/en/comps/10/2023-2024/stats/2023-2024-Championship-Stats'
        
        driver = None
        for attempt in range(retry_count):
            try:
                if driver:
                    driver.quit()
                driver = self.get_driver()
                
                self.logger.info(f"Scraping Championship 2023-24 standard stats (Attempt {attempt + 1}/{retry_count})")
                self.logger.info(f"URL: {url}")
                
                driver.get(url)
                
                # Wait for table to load
                if not self.wait_for_table(driver):
                    self.logger.warning(f"Table not found after waiting")
                    continue
                
                # Add delay to ensure full page load
                time.sleep(random.uniform(5, 8))
                
                # Get page source and parse
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                table = self.find_table(soup)
                
                if not table:
                    self.logger.error(f"Table not found in page source")
                    continue
                
                tbody = table.find('tbody')
                if not tbody:
                    self.logger.error(f"No tbody found in table")
                    continue
                
                # Extract player data
                data = []
                rows = tbody.find_all('tr')
                self.logger.info(f"Found {len(rows)} rows in table")
                
                for i, row in enumerate(rows):
                    # Skip header rows
                    if 'thead' in row.get('class', []):
                        continue
                    
                    player_data = self.extract_player_data(row)
                    if player_data:
                        data.append(player_data)
                    
                    if i % 50 == 0:
                        self.logger.info(f"Processed {i} rows...")
                
                if not data:
                    self.logger.error(f"No data rows extracted")
                    continue
                
                # Create DataFrame
                df = pd.DataFrame(data)
                
                # Log what we found
                self.logger.info(f"Successfully extracted {len(df)} players")
                self.logger.info(f"Columns: {list(df.columns)}")
                
                # Verify expected columns
                expected_cols = set(self.standard_stats['expected_columns'])
                actual_cols = set(df.columns)
                missing_cols = expected_cols - actual_cols
                
                if missing_cols:
                    self.logger.warning(f"Missing columns: {missing_cols}")
                    # Add missing columns with empty values
                    for col in missing_cols:
                        df[col] = ''
                
                return df
                
            except Exception as e:
                self.logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt < retry_count - 1:
                    self.logger.info(f"Waiting before retry...")
                    time.sleep(random.uniform(10, 15))
                continue
            finally:
                if driver:
                    driver.quit()
        
        self.logger.error("All attempts failed")
        return pd.DataFrame()

    def clean_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric data while preserving important info"""
        non_numeric_cols = ['player', 'nationality', 'position', 'team', 'player_id']
        
        for col in df.columns:
            if col not in non_numeric_cols:
                try:
                    # Remove any remaining percentage signs and convert to numeric
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str).str.replace('%', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    continue
        
        return df

    def save_data(self, df: pd.DataFrame) -> bool:
        """Save the scraped data"""
        try:
            # Create directory
            season_dir = os.path.join(self.data_dir, 'Championship_2023-2024')
            os.makedirs(season_dir, exist_ok=True)
            
            # Clean numeric data
            df = self.clean_numeric_data(df)
            
            # Save to CSV
            filepath = os.path.join(season_dir, 'standard_stats.csv')
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"Successfully saved {len(df)} players to {filepath}")
            
            # Log some sample data
            self.logger.info("\nSample of scraped data:")
            self.logger.info(df[['player', 'team', 'goals', 'assists', 'minutes']].head().to_string())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            return False


def main():
    """Main function to scrape Championship 2023-24 data"""
    print("Championship 2023-24 Standard Stats Scraper")
    print("=" * 50)
    
    scraper = ChampionshipScraper()
    
    try:
        # Scrape the data
        df = scraper.scrape_championship_2023_24()
        
        if df.empty:
            print("Failed to scrape data. Check logs for details.")
            return False
        
        # Save the data
        success = scraper.save_data(df)
        
        if success:
            print(f"\nSUCCESS! Scraped {len(df)} Championship players from 2023-24 season")
            print(f"Data saved to: data/raw/Championship_2023-2024/standard_stats.csv")
            print("\nYou can now re-run your analysis with complete data!")
            return True
        else:
            print("Failed to save data. Check logs for details.")
            return False
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\nNext steps:")
        print("1. Re-run your equivalency analysis")
        print("2. Update your visualizations")
        print("3. Update your paper with the complete dataset!")
    else:
        print("\nTroubleshooting:")
        print("1. Ensure Chrome/Chromium is installed")
        print("2. Install chromedriver: pip install chromedriver-autoinstaller")
        print("3. Check the logs for detailed error messages")
