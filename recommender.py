"""
Recommendation System using Cornac
Recommends top items for each user
"""
import numpy as np
from cornac.models import MF, BPR, NMF
from data_loader import RecommendationDataLoader


class RecommenderSystem:
    """Recommendation System using Cornac"""
    
    def __init__(self, dataset, model_name='MF', **kwargs):
        """
        Initialize recommender system
        
        Args:
            dataset: Cornac Dataset object
            model_name (str): Model to use - 'MF' (Matrix Factorization), 'BPR', 'NMF'
            **kwargs: Additional arguments for model
        """
        self.dataset = dataset
        self.model_name = model_name
        self.model = None
        self.kwargs = kwargs
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the recommendation model"""
        
        # Default parameters for models
        mf_params = {
            'k': self.kwargs.get('k', 10),  # latent dimensions
            'learning_rate': self.kwargs.get('learning_rate', 0.01),
            'lambda_reg': self.kwargs.get('lambda_reg', 0.01),  # regularization
            'max_iter': self.kwargs.get('max_iter', 100),
            'verbose': self.kwargs.get('verbose', True)
        }
        
        bpr_params = {
            'k': self.kwargs.get('k', 10),
            'learning_rate': self.kwargs.get('learning_rate', 0.01),
            'lambda_reg': self.kwargs.get('lambda_reg', 0.01),
            'max_iter': self.kwargs.get('max_iter', 100),
            'verbose': self.kwargs.get('verbose', True)
        }
        
        nmf_params = {
            'k': self.kwargs.get('k', 10),
            'learning_rate': self.kwargs.get('learning_rate', 0.01),
            'lambda_reg': self.kwargs.get('lambda_reg', 0.01),
            'max_iter': self.kwargs.get('max_iter', 100),
            'verbose': self.kwargs.get('verbose', True)
        }
        
        if self.model_name == 'MF':
            self.model = MF(**mf_params)
            print(f"✓ Initialized Matrix Factorization model")
            
        elif self.model_name == 'BPR':
            self.model = BPR(**bpr_params)
            print(f"✓ Initialized BPR model")
            
        elif self.model_name == 'NMF':
            self.model = NMF(**nmf_params)
            print(f"✓ Initialized NMF model")
            
        else:
            # Default to MF
            self.model = MF(**mf_params)
            print(f"✓ Model '{self.model_name}' not recognized, using Matrix Factorization")
    
    def fit(self):
        """Train the recommendation model"""
        print(f"\n{'='*50}")
        print(f"Training {self.model_name} model...")
        print(f"{'='*50}")
        
        self.model.fit(self.dataset)
        print(f"✓ Model training completed")
    
    def recommend(self, user_id, num_recommendations=10):
        """
        Get top recommendations for a user
        
        Args:
            user_id: User identifier
            num_recommendations (int): Number of items to recommend
        
        Returns:
            list: List of (item_id, score) tuples
        """
        # Get user internal ID
        user_iid = self.dataset.get_user(user_id).iid if self.dataset.get_user(user_id) else None
        
        if user_iid is None:
            print(f"Warning: User {user_id} not found in dataset")
            return []
        
        # Get scores for all items
        item_scores = []
        for item in self.dataset.items:
            item_iid = item.iid
            score = self.model.score(user_iid, item_iid)
            item_scores.append((item.uid, score))  # item.uid is the original item_id
        
        # Sort by score and get top N
        item_scores.sort(key=lambda x: x[1], reverse=True)
        top_items = item_scores[:num_recommendations]
        
        return top_items
    
    def recommend_all_users(self, num_recommendations=10):
        """
        Get top recommendations for all users
        
        Args:
            num_recommendations (int): Number of items to recommend per user
        
        Returns:
            dict: Dictionary with user_id as key and list of recommended items as value
        """
        recommendations = {}
        
        print(f"\nGenerating recommendations for all users...")
        
        # user_ids is a list where index i gives the original user_id for internal index i
        # item_ids is a list where index j gives the original item_id for internal index j
        user_ids = self.dataset.user_ids
        item_ids = self.dataset.item_ids
        
        # Get all users
        num_users = self.dataset.num_users
        
        for user_iid in range(num_users):
            # Get the original user ID
            user_id = user_ids[user_iid]
            
            # Get scores for all items
            item_scores = []
            num_items = self.dataset.num_items
            
            for item_iid in range(num_items):
                try:
                    score = self.model.score(user_iid, item_iid)
                    item_id = item_ids[item_iid]
                    item_scores.append((item_id, score))
                except:
                    continue
            
            # Sort by score and get top N
            item_scores.sort(key=lambda x: x[1], reverse=True)
            top_items = [str(item_id) for item_id, score in item_scores[:num_recommendations]]
            
            recommendations[str(user_id)] = top_items
        
        print(f"✓ Generated recommendations for {len(recommendations)} users")
        
        return recommendations
    
    def save_recommendations_to_file(self, recommendations, output_file):
        """
        Save recommendations to file (one user per line with recommended items)
        
        Args:
            recommendations (dict): Recommendation dictionary from recommend_all_users()
            output_file (str): Output file path
        """
        with open(output_file, 'w') as f:
            for user_id in sorted(recommendations.keys(), key=lambda x: int(x) if x.isdigit() else x):
                items = recommendations[user_id]
                line = f"{user_id} " + " ".join(str(item) for item in items)
                f.write(line + "\n")
        
        print(f"✓ Saved recommendations to {output_file}")
