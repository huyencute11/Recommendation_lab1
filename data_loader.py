"""
Data Loader Module for Recommendation System
Loads training data from CSV file with columns: user_id, item_id, rating
"""
import numpy as np
import pandas as pd
from cornac.data import Dataset
from cornac.data import Reader


class RecommendationDataLoader:
    """Load and preprocess recommendation data"""
    
    def __init__(self, data_file_path):
        """
        Initialize data loader
        
        Args:
            data_file_path (str): Path to training data file (CSV format)
        """
        self.data_file_path = data_file_path
        self.data = None
        self.dataset = None
    
    def load_data(self):
        """
        Load data from CSV file.
        Expected format: CSV with columns: user_id, item_id, rating
        """
        try:
            # Load CSV file
            self.data = pd.read_csv(self.data_file_path)
            
            # Validate columns
            required_columns = ['user_id', 'item_id', 'rating']
            if not all(col in self.data.columns for col in required_columns):
                print(f"Error: CSV must contain columns: {required_columns}")
                print(f"Found columns: {list(self.data.columns)}")
                return None
            
            # Convert to correct data types
            self.data['user_id'] = self.data['user_id'].astype(str)
            self.data['item_id'] = self.data['item_id'].astype(str)
            self.data['rating'] = pd.to_numeric(self.data['rating'], errors='coerce')
            
            # Remove rows with NaN ratings
            self.data = self.data.dropna(subset=['rating'])
            
            print(f"✓ Loaded {len(self.data)} ratings")
            print(f"  • Users: {len(self.data['user_id'].unique())}")
            print(f"  • Items: {len(self.data['item_id'].unique())}")
            print(f"  • Rating range: {self.data['rating'].min():.1f} - {self.data['rating'].max():.1f}")
            
            return self.data
            
        except FileNotFoundError:
            print(f"Error: File not found at {self.data_file_path}")
            return None
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    
    def create_dataset(self, fmt='UIR'):
        """
        Create cornac Dataset from loaded data
        
        Args:
            fmt (str): Data format - 'UIR' (User-Item-Rating) by default
        
        Returns:
            cornac.data.Dataset: Dataset object for training
        """
        if self.data is None:
            self.load_data()
        
        if self.data is None:
            return None
        
        # Convert to list of tuples for cornac format
        # cornac expects: (user_id, item_id, rating)
        uir_list = self.data[['user_id', 'item_id', 'rating']].values.tolist()
        
        # Create dataset using from_uir
        self.dataset = Dataset.from_uir(uir_list)
        
        print(f"✓ Dataset created with {self.dataset.num_users} users and {self.dataset.num_items} items")
        
        return self.dataset
    
    def get_statistics(self):
        """Get data statistics"""
        if self.data is None:
            self.load_data()
        
        if self.data is None:
            return None
        
        stats = {
            'num_users': len(self.data['user_id'].unique()),
            'num_items': len(self.data['item_id'].unique()),
            'num_ratings': len(self.data),
            'sparsity': 1 - (len(self.data) / (len(self.data['user_id'].unique()) * len(self.data['item_id'].unique())))
        }
        
        return stats
