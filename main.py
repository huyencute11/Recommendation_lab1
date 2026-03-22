"""
Main entry point for the Recommendation System
"""
import sys
from data_loader import RecommendationDataLoader
from recommender import RecommenderSystem


def main():
    """
    Main function to run the recommendation system
    
    Usage:
        python main.py <training_data_file> [model_name] [num_recommendations]
    
    Example:
        python main.py training_data.txt MF 10
    """
    
    # Get arguments
    if len(sys.argv) < 2:
        print("Usage: python main.py <training_data_file> [model_name] [num_recommendations]")
        print("\nExample: python main.py training_data.txt MF 10")
        print("\nModel options: 'MF' (Matrix Factorization), 'BPR', 'NMF'")
        sys.exit(1)
    
    training_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'MF'
    num_recommendations = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    print(f"{'='*60}")
    print(f"Recommendation System using Cornac")
    print(f"{'='*60}\n")
    
    # Load data
    print(f"Loading data from: {training_file}")
    loader = RecommendationDataLoader(training_file)
    loader.load_data()
    
    # Print statistics
    stats = loader.get_statistics()
    if stats:
        print(f"\nData Statistics:")
        print(f"  • Users: {stats['num_users']}")
        print(f"  • Items: {stats['num_items']}")
        print(f"  • Ratings: {stats['num_ratings']}")
        print(f"  • Sparsity: {stats['sparsity']:.4f}")
    
    # Create dataset
    dataset = loader.create_dataset()
    if dataset is None:
        print("Error: Failed to create dataset")
        sys.exit(1)
    
    # Initialize and train recommender
    recommender = RecommenderSystem(
        dataset,
        model_name=model_name,
        k=10,
        learning_rate=0.001,
        lambda_u=0.01,
        lambda_i=0.01,
        max_iter=100,
        verbose=True
    )
    
    recommender.fit()
    
    # Get recommendations for all users
    recommendations = recommender.recommend_all_users(num_recommendations=num_recommendations)
    
    # Display sample recommendations
    print(f"\n{'='*60}")
    print(f"Sample Recommendations (showing first 5 users):")
    print(f"{'='*60}")
    
    user_ids = sorted(recommendations.keys(), key=lambda x: int(x) if x.isdigit() else x)
    for user_id in user_ids[:5]:
        items = recommendations[user_id]
        print(f"User {user_id}: {' '.join(str(item) for item in items)}")
    
    # Save all recommendations to file
    if training_file.endswith('.csv'):
        output_file = training_file.replace('.csv', '_recommendations.txt')
    else:
        output_file = training_file.replace('.txt', '_recommendations.txt')
    recommender.save_recommendations_to_file(recommendations, output_file)
    
    print(f"\n{'='*60}")
    print(f"Recommendation process completed!")
    print(f"Output file: {output_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
