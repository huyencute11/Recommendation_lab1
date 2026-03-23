#!/bin/bash

# Script to test all models and find the best one

echo "============================================================"
echo "Testing All Recommendation Models"
echo "============================================================"

cd /Volumes/DATA/Master/recommendation_system/code_trainData
source venv/bin/activate

echo ""
echo "Recommended order to try (best first):"
echo "1. PMF (Probabilistic Matrix Factorization) - BEST for ratings"
echo "2. BPR (Bayesian Personalized Ranking) - BEST for implicit feedback"
echo "3. MF (Matrix Factorization) - Classic collaborative filtering"
echo "4. NMF (Non-negative MF) - For count data"
echo ""

# Test PMF (RECOMMENDED FIRST)
echo "============================================================"
echo "Testing PMF (Probabilistic Matrix Factorization)..."
echo "============================================================"
rm -f train_v3_recommendations.txt
python3 main.py train_v3.csv PMF 50
echo "✓ PMF done! Output: train_v3_recommendations.txt"
echo "  Submit this to Codabench and check results."
echo ""

# Uncomment to test BPR
# echo "============================================================"
# echo "Testing BPR (Bayesian Personalized Ranking)..."
# echo "============================================================"
# rm -f train_v3_recommendations.txt
# python3 main.py train_v3.csv BPR 50
# echo "✓ BPR done!"
# echo ""

# Uncomment to test MF
# echo "============================================================"
# echo "Testing MF (Matrix Factorization)..."
# echo "============================================================"
# rm -f train_v3_recommendations.txt
# python3 main.py train_v3.csv MF 50
# echo "✓ MF done!"
# echo ""

echo "============================================================"
echo "Instructions:"
echo "1. Run this script: bash test_models.sh"
echo "2. Each model will generate train_v3_recommendations.txt"
echo "3. Submit each to Codabench (one at a time)"
echo "4. Compare results and use the best model"
echo "============================================================"
