# Recommendation System using Cornac

A Python-based recommendation system that uses the Cornac library to recommend top products for users.

## Features

- **Matrix Factorization (MF)**: Traditional collaborative filtering approach
- **BPR (Bayesian Personalized Ranking)**: Pairwise ranking for implicit feedback
- **NMF (Non-negative Matrix Factorization)**: Non-negative matrix factorization approach
- Flexible configuration for model parameters
- Efficient batch recommendations for all users

## File Structure

```
.
├── requirements.txt          # Python dependencies
├── data_loader.py           # Data loading and preprocessing module
├── recommender.py           # Recommendation system module
├── main.py                  # Main entry point
├── README.md               # This file
└── training_data.txt       # Your training data (to be provided)
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

Your training data should be in the following format:

```
person_id item_id_1 item_id_2 ... item_id_50
person_id item_id_1 item_id_2 ... item_id_50
...
```

**Example:**
```
0 1 5 12 23 45 67 89 102 154 ...
1 2 8 15 34 56 78 90 111 200 ...
2 3 9 18 42 62 84 105 123 234 ...
```

- Each line starts with a person/user ID
- Followed by the item IDs that this person has interacted with (50 items in your case)
- Spaces separate the IDs

## Usage

### Basic Usage

```bash
python main.py training_data.txt
```

This will:
- Load the training data
- Train a Matrix Factorization model by default
- Generate top 10 recommendations for each user
- Save results to `training_data_recommendations.txt`

### Advanced Usage

```bash
python main.py <training_file> <model_name> <num_recommendations>
```

**Parameters:**
- `<training_file>`: Path to your training data file
- `<model_name>`: Model algorithm ('MF', 'BPR', or 'NMF')
- `<num_recommendations>`: Number of items to recommend per user

**Examples:**

```bash
# Use BPR model and get top 15 recommendations
python main.py training_data.txt BPR 15

# Use NMF model and get top 20 recommendations
python main.py training_data.txt NMF 20
```

## Output

The output file will be in the same format as the input:

```
person_id item_id_1 item_id_2 ... item_id_N
person_id item_id_1 item_id_2 ... item_id_N
...
```

Where each line contains the user ID followed by their top N recommended item IDs.

## Model Parameters

You can modify model parameters in `main.py`:

- **k**: Number of latent dimensions (default: 10)
- **learning_rate**: Learning rate for optimization (default: 0.001)
- **lambda_u**: Regularization for user factors (default: 0.01)
- **lambda_i**: Regularization for item factors (default: 0.01)
- **max_iter**: Number of training iterations (default: 100)

Adjust these values in the `RecommenderSystem` initialization based on your needs.

## Example Workflow

1. Prepare your training data in the required format
2. Place it in this directory (e.g., `training_data.txt`)
3. Run the system:
   ```bash
   python main.py training_data.txt MF 10
   ```
4. Check the output file: `training_data_recommendations.txt`

## Troubleshooting

- **Module not found**: Make sure all dependencies are installed with `pip install -r requirements.txt`
- **Data format error**: Verify your training data follows the required format (space-separated IDs)
- **File not found**: Check that the training data file path is correct

## Algorithm Explanations

### Matrix Factorization (MF)
Decomposes the user-item interaction matrix into two lower-rank matrices representing user and item latent factors. Good for dense interactions.

### BPR (Bayesian Personalized Ranking)
Optimized for implicit feedback (binary interactions). Learns from user preferences through pairwise comparisons.

### NMF (Non-negative Matrix Factorization)
Similar to MF but ensures non-negative factors, which can be more interpretable. Good for count-based interactions.

## Requirements

- Python 3.7+
- cornac >= 1.3.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- scikit-learn >= 0.24.0
