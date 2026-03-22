"""
Quick Start Guide for Testing the Recommendation System
"""

# Step 1: Install dependencies
# Run in terminal:
# cd /Volumes/DATA/Master/recommendation_system/code_trainData
# pip install -r requirements.txt

# Step 2: Test with sample data
# python main.py sample_training_data.txt MF 10

# Step 3: When you have your own training data, run:
# python main.py your_training_data.txt MF 10

# All available models:
# - MF (Matrix Factorization) - Default, good for general use
# - BPR (Bayesian Personalized Ranking) - Better for implicit feedback (binary interactions)
# - NMF (Non-negative Matrix Factorization) - Better for count-based data

# Example commands:
# ================

# Test with default settings (Matrix Factorization, top 10 recommendations):
# python main.py sample_training_data.txt

# Use BPR model with top 15 recommendations:
# python main.py sample_training_data.txt BPR 15

# Use NMF model with top 20 recommendations:
# python main.py sample_training_data.txt NMF 20

# When you provide your training data file "my_data.txt":
# python main.py my_data.txt MF 10
# python main.py my_data.txt BPR 10
# python main.py my_data.txt NMF 10

# Output:
# ======
# The program will create a file: sample_training_data_recommendations.txt
# Or: your_training_data_recommendations.txt
# 
# Format: Each line contains:
# user_id recommended_item_1 recommended_item_2 ... recommended_item_N

# To customize model parameters, edit main.py and modify the RecommenderSystem initialization
