"""
Optimized BPR Recommendation System with implicit feedback and negative sampling
"""
import sys
from enhanced_data_loader import EnhancedDataLoader
from cornac.models import BPR
from cornac.data import Dataset


def main():
    """Run optimized BPR recommendation system"""
    
    if len(sys.argv) < 2:
        print("Usage: python bpr_optimized.py <training_data_file> [num_recommendations]")
        print("\nExample: python bpr_optimized.py train_v3.csv 50")
        sys.exit(1)
    
    training_file = sys.argv[1]
    num_recommendations = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    print(f"{'='*60}")
    print(f"Optimized BPR Recommendation System")
    print(f"{'='*60}\n")
    
    # Load data with implicit feedback conversion
    print(f"Loading data from: {training_file}")
    loader = EnhancedDataLoader(training_file, rating_threshold=4.0)  # ratings >= 4 are positive
    loader.load_data()
    
    # Create implicit feedback dataset
    dataset = loader.create_dataset_implicit()
    if dataset is None:
        print("Error: Failed to create dataset")
        sys.exit(1)
    
    # Initialize BPR with optimized hyperparameters
    # BPR hyperparameters tuned for negative sampling and ranking
    print(f"\n{'='*60}")
    print(f"Training BPR (Bayesian Personalized Ranking)...")
    print(f"{'='*60}")
    print(f"  • k (latent dimensions): 100")
    print(f"  • learning_rate: 0.01")
    print(f"  • lambda_reg: 0.001")
    print(f"  • max_iter: 200")
    print(f"  • neg_sampling: enabled (negative sampling)")
    
    bpr = BPR(
        k=100,                    # More latent dimensions for better representation
        learning_rate=0.01,       # Slower learning for stability
        lambda_reg=0.001,         # Low regularization for flexibility
        max_iter=200,             # More iterations
        seed=123,                 # For reproducibility
        verbose=True
    )
    
    # Train model
    bpr.fit(dataset)
    print(f"\n✓ Model training completed")
    
    # Generate recommendations for all users
    print(f"\nGenerating recommendations for all users...")
    
    recommendations = {}
    user_ids = dataset.user_ids
    item_ids = dataset.item_ids
    num_users = dataset.num_users
    num_items = dataset.num_items
    
    for user_iid in range(num_users):
        # Get scores for all items
        item_scores = []
        
        for item_iid in range(num_items):
            try:
                score = bpr.score(user_iid, item_iid)
                item_id = item_ids[item_iid]
                item_scores.append((item_id, score))
            except:
                continue
        
        # Sort by score and get top N
        item_scores.sort(key=lambda x: x[1], reverse=True)
        top_items = [str(item_id) for item_id, score in item_scores[:num_recommendations]]
        
        recommendations[user_iid] = top_items
    
    print(f"✓ Generated recommendations for {len(recommendations)} users")
    
    # Display sample recommendations
    print(f"\n{'='*60}")
    print(f"Sample Recommendations (first 5 users):")
    print(f"{'='*60}")
    
    for user_id in sorted(recommendations.keys())[:5]:
        items = recommendations[user_id]
        print(f"User {user_id}: {' '.join(str(item) for item in items[:10])}...")
    
    # Save recommendations
    output_file = training_file.replace('.csv', '_recommendations.txt')
    
    with open(output_file, 'w') as f:
        num_users_output = len(recommendations)
        for user_iid in range(num_users_output):
            if user_iid in recommendations:
                items = recommendations[user_iid]
                line = " ".join(str(item) for item in items)
                f.write(line + "\n")
    
    print(f"\n{'='*60}")
    print(f"✓ Saved recommendations to: {output_file}")
    print(f"  • Format: Items sorted by ranking score (descending)")
    print(f"  • Line index = user index (0, 1, 2, ...)")
    print(f"  • Each line = top {num_recommendations} recommendations")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
