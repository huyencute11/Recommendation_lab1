#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Run recommendation system with your training data
# Format: python3 main.py <csv_file> <model_name> <num_recommendations>

echo "Running Recommendation System..."
echo "================================"

# Example with train_v3.csv, Matrix Factorization model, top 50 recommendations
python3 main.py train_v3.csv MF 50

echo "================================"
echo "Done! Check train_v3_recommendations.txt for results"
