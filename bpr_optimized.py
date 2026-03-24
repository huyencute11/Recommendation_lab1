"""
Optimized BPR Recommendation System with implicit feedback and negative sampling
"""
import argparse
import sys
from collections import defaultdict
from enhanced_data_loader import EnhancedDataLoader
from cornac.models import BPR


def parse_args():
    parser = argparse.ArgumentParser(description="Train BPR and export submission file.")
    parser.add_argument("training_data_file", type=str, help="Path to CSV training file")
    parser.add_argument("num_recommendations", type=int, nargs="?", default=50, help="Top-N items per user")
    parser.add_argument("--rating-threshold", type=float, default=3.5, help="Positive rating threshold")
    parser.add_argument("--latent-dim", type=int, default=100, help="BPR latent dimension")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="BPR learning rate")
    parser.add_argument("--lambda-reg", type=float, default=0.001, help="BPR regularization")
    parser.add_argument("--max-iter", type=int, default=200, help="BPR max iterations")
    return parser.parse_args()


def main():
    """Run optimized BPR recommendation system"""
    args = parse_args()
    training_file = args.training_data_file
    num_recommendations = args.num_recommendations
    
    print(f"{'='*60}")
    print(f"Optimized BPR Recommendation System")
    print(f"{'='*60}\n")
    
    # Load data with implicit feedback conversion
    print(f"Loading data from: {training_file}")
    loader = EnhancedDataLoader(training_file, rating_threshold=args.rating_threshold, ensure_all_users=True)
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
    print(f"  • rating_threshold: {args.rating_threshold}")
    print(f"  • k (latent dimensions): {args.latent_dim}")
    print(f"  • learning_rate: {args.learning_rate}")
    print(f"  • lambda_reg: {args.lambda_reg}")
    print(f"  • max_iter: {args.max_iter}")
    print(f"  • neg_sampling: enabled (negative sampling)")
    
    bpr = BPR(
        k=args.latent_dim,
        learning_rate=args.learning_rate,
        lambda_reg=args.lambda_reg,
        max_iter=args.max_iter,
        seed=123,
        verbose=True
    )
    
    # Train model
    bpr.fit(dataset)
    print(f"\n✓ Model training completed")
    
    # Generate recommendations for all users
    print(f"\nGenerating recommendations for all users...")
    
    user_ids = dataset.user_ids
    item_ids = dataset.item_ids
    raw_df = loader.data.copy()
    recommendations = {}
    num_users = dataset.num_users
    num_items = dataset.num_items

    user_seen_items = defaultdict(set)
    for _, row in raw_df.iterrows():
        user_seen_items[int(row["user_id"])].add(int(row["item_id"]))

    implicit_df = loader.convert_to_implicit_feedback()
    item_popularity = implicit_df["item_id"].value_counts().to_dict()
    popularity_ranked_items = [int(item) for item, _ in sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)]
    
    for user_iid in range(num_users):
        item_scores = []
        user_id = int(user_ids[user_iid])
        seen_items = user_seen_items.get(user_id, set())
        
        for item_iid in range(num_items):
            try:
                score = bpr.score(user_iid, item_iid)
                item_id = int(item_ids[item_iid])
                if item_id in seen_items:
                    continue
                pop_boost = item_popularity.get(str(item_id), 0)
                item_scores.append((item_id, score + 1e-6 * pop_boost))
            except Exception:
                continue
        
        item_scores.sort(key=lambda x: x[1], reverse=True)
        top_items = [str(item_id) for item_id, _ in item_scores[:num_recommendations]]

        if len(top_items) < num_recommendations:
            for popular_item in popularity_ranked_items:
                if popular_item in seen_items:
                    continue
                if str(popular_item) in top_items:
                    continue
                top_items.append(str(popular_item))
                if len(top_items) == num_recommendations:
                    break

        recommendations[user_id] = top_items
    
    print(f"✓ Generated recommendations for {len(recommendations)} users")
    
    # Display sample recommendations
    print(f"\n{'='*60}")
    print(f"Sample Recommendations (first 5 users):")
    print(f"{'='*60}")
    
    for user_id in sorted(recommendations.keys())[:5]:
        items = recommendations[user_id]
        print(f"User {user_id}: {' '.join(str(item) for item in items[:10])}...")
    
    min_user_id = int(raw_df["user_id"].astype(int).min())
    max_user_id = int(raw_df["user_id"].astype(int).max())
    output_file = training_file.replace('.csv', '_recommendations.txt')
    
    with open(output_file, 'w') as f:
        for user_id in range(min_user_id, max_user_id + 1):
            items = recommendations.get(user_id)
            if not items:
                seen_items = user_seen_items.get(user_id, set())
                items = []
                for popular_item in popularity_ranked_items:
                    if popular_item in seen_items:
                        continue
                    items.append(str(popular_item))
                    if len(items) == num_recommendations:
                        break
            f.write(" ".join(items) + "\n")
    
    print(f"\n{'='*60}")
    print(f"✓ Saved recommendations to: {output_file}")
    print(f"  • Format: Items sorted by ranking score (descending)")
    print(f"  • Line order: ascending original user_id ({min_user_id}..{max_user_id})")
    print(f"  • Total lines: {max_user_id - min_user_id + 1}")
    print(f"  • Each line = top {num_recommendations} recommendations")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
