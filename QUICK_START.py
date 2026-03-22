"""
Quick Start Guide for Testing the Recommendation System
"""

# Step 0: First Time Setup
# ========================
# Option A: Automatic Setup (interactive)
#   python3 setup_and_run.py
#
# Option B: Manual Setup
#   python3 -m venv venv
#   source venv/bin/activate (on macOS/Linux) or venv\Scripts\activate (Windows)
#   pip install -r requirements.txt

# Step 1: Activate Virtual Environment (if not already activated)
# ================================================================
# macOS/Linux:
#   source venv/bin/activate
#
# Windows Command Prompt:
#   venv\Scripts\activate
#
# Windows PowerShell:
#   venv\Scripts\Activate.ps1

# Step 2: Run Recommendation System with Your Data
# =================================================
# Format: python3 main.py <csv_file> <model> <num_recommendations>

# Example Commands:
# ================

# Basic - Top 50 recommendations with Matrix Factorization (default)
# python3 main.py train_v3.csv MF 50

# Top 20 recommendations with BPR (better for implicit feedback)
# python3 main.py train_v3.csv BPR 20

# Top 30 recommendations with NMF
# python3 main.py train_v3.csv NMF 30

# Available Models:
# ================
# - MF (Matrix Factorization)
#   • Best for: General-purpose, balanced performance
#   • Use when: You have explicit ratings
#
# - BPR (Bayesian Personalized Ranking)
#   • Best for: Implicit feedback (binary interactions)
#   • Use when: Data is only positive interactions (clicks, purchases)
#
# - NMF (Non-negative Matrix Factorization)
#   • Best for: Count-based or non-negative interactions
#   • Use when: Data represents counts or quantities

# Output:
# ======
# Input: train_v3.csv
# Output: train_v3_recommendations.txt
#
# Output Format:
# ==============
# user_id recommended_item_1 recommended_item_2 ... recommended_item_N
# Example:
# 1494 234 567 890 123 456 789 ...
# 9808 345 678 901 234 567 890 ...
# 19854 456 789 12 345 678 901 ...

# CSV Data Format Required:
# =========================
# Your CSV must have 3 columns:
# user_id,item_id,rating
# 1494,1,4
# 1494,2,5
# 9808,3,1
# 9808,4,3

# Troubleshooting:
# ===============
# "Module not found" error?
#   → Make sure virtual environment is activated
#   → Run: pip install -r requirements.txt
#
# CSV format error?
#   → Check your CSV has exactly: user_id, item_id, rating
#
# File not found?
#   → Verify CSV filename and it's in the correct directory

