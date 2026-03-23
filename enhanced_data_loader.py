"""
Enhanced Data Loader with implicit feedback conversion and negative sampling
"""
import numpy as np
import pandas as pd
from cornac.data import Dataset


class EnhancedDataLoader:
    """Enhanced loader with implicit feedback and negative sampling"""
    
    def __init__(self, data_file_path, rating_threshold=3.0):
        """
        Initialize enhanced data loader
        
        Args:
            data_file_path (str): Path to CSV training data
            rating_threshold (float): Ratings >= threshold are positive interactions
        """
        self.data_file_path = data_file_path
        self.rating_threshold = rating_threshold
        self.data = None
        self.dataset = None
    
    def load_data(self):
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(self.data_file_path)
            
            # Validate columns
            required_columns = ['user_id', 'item_id', 'rating']
            if not all(col in self.data.columns for col in required_columns):
                print(f"Error: CSV must contain columns: {required_columns}")
                return None
            
            # Convert to correct data types
            self.data['user_id'] = self.data['user_id'].astype(str)
            self.data['item_id'] = self.data['item_id'].astype(str)
            self.data['rating'] = pd.to_numeric(self.data['rating'], errors='coerce')
            
            # Remove rows with NaN ratings
            self.data = self.data.dropna(subset=['rating'])
            
            print(f"✓ Loaded {len(self.data)} ratings (raw)")
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
    
    def convert_to_implicit_feedback(self):
        """
        Convert explicit ratings to implicit feedback (binary)
        rating >= threshold = positive (1), else = negative (0)
        """
        if self.data is None:
            self.load_data()
        
        if self.data is None:
            return None
        
        # Create implicit feedback: 1 for positive, remove negatives
        self.data['implicit'] = (self.data['rating'] >= self.rating_threshold).astype(int)
        
        # Keep only positive interactions
        implicit_data = self.data[self.data['implicit'] == 1].copy()
        
        print(f"\n✓ Converted to implicit feedback (rating >= {self.rating_threshold}):")
        print(f"  • Kept interactions: {len(implicit_data)}")
        print(f"  • Users: {len(implicit_data['user_id'].unique())}")
        print(f"  • Items: {len(implicit_data['item_id'].unique())}")
        print(f"  • Sparsity: {1 - (len(implicit_data) / (len(implicit_data['user_id'].unique()) * len(implicit_data['item_id'].unique()))):.4f}")
        
        return implicit_data
    
    def create_dataset_implicit(self):
        """
        Create cornac Dataset from implicit feedback
        Uses only positive interactions (no explicit ratings)
        """
        implicit_data = self.convert_to_implicit_feedback()
        
        if implicit_data is None:
            return None
        
        # Convert to list of tuples for cornac format (user_id, item_id)
        # For implicit feedback, we only need (user, item) pairs
        uir_list = implicit_data[['user_id', 'item_id']].values.tolist()
        
        # Create dataset using from_uir - implicit feedback
        self.dataset = Dataset.from_uir(uir_list)
        
        print(f"\n✓ Dataset created for BPR:")
        print(f"  • Users: {self.dataset.num_users}")
        print(f"  • Items: {self.dataset.num_items}")
        print(f"  • Interactions: {self.dataset.num_ratings}")
        
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
