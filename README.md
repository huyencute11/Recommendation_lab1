# Recommendation System using Cornac

A Python-based recommendation system that uses the Cornac library to recommend top products for users based on user-item-rating data.

## Features

- **Matrix Factorization (MF)**: Traditional collaborative filtering approach
- **BPR (Bayesian Personalized Ranking)**: Pairwise ranking for implicit feedback
- **NMF (Non-negative Matrix Factorization)**: Non-negative matrix factorization approach
- Flexible configuration for model parameters
- Efficient batch recommendations for all users
- CSV data format support

## File Structure

```
.
├── requirements.txt          # Python dependencies
├── data_loader.py           # Data loading and preprocessing module
├── recommender.py           # Recommendation system module
├── main.py                  # Main entry point
├── RUN.sh                   # Quick run script
├── README.md               # This file
└── train_v3.csv            # Your training data (CSV format)
```

## Installation

1. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Format

Your training data should be a CSV file with 3 columns: **user_id**, **item_id**, **rating**

**Example:**
```
user_id,item_id,rating
1494,1,4
1494,2,5
9808,3,1
9808,4,3
19854,5,5
```

- **user_id**: ID of the user/person
- **item_id**: ID of the product/item
- **rating**: Rating/score (numeric value)

## Usage

### Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run with your CSV data, Matrix Factorization model, top 50 recommendations
python3 main.py train_v3.csv MF 50
```

Or use the provided script:
```bash
chmod +x RUN.sh
./RUN.sh
```

### Full Usage

```bash
python3 main.py <csv_file> <model_name> <num_recommendations>
```

**Parameters:**
- `<csv_file>`: Path to your CSV training data file
- `<model_name>`: Model algorithm ('MF', 'BPR', or 'NMF')
- `<num_recommendations>`: Number of items to recommend per user (default: 10)

**Examples:**

```bash
# Use BPR model and get top 15 recommendations
python3 main.py train_v3.csv BPR 15

# Use NMF model and get top 50 recommendations
python3 main.py train_v3.csv NMF 50

# Use default (MF model) with top 20 recommendations
python3 main.py train_v3.csv MF 20
```

## Output

The output file will be generated as: `<input_file>_recommendations.txt`

**Format:** Each line contains the user ID followed by their top N recommended item IDs

**Example output (train_v3_recommendations.txt):**
```
1494 234 567 890 123 456 ...
9808 345 678 901 234 567 ...
19854 456 789 12 345 678 ...
```

Where each number is a product/item ID recommended for that user.

## Model Parameters

You can modify model parameters in `main.py`:

- **k**: Number of latent dimensions (default: 10)
- **learning_rate**: Learning rate for optimization (default: 0.001)
- **lambda_u**: Regularization for user factors (default: 0.01)
- **lambda_i**: Regularization for item factors (default: 0.01)
- **max_iter**: Number of training iterations (default: 100)

Adjust these values in the `RecommenderSystem` initialization based on your needs.

## Example Workflow

1. Ensure your training data is in CSV format with columns: `user_id`, `item_id`, `rating`
2. Place your CSV file in this directory (e.g., `train_v3.csv`)
3. Run the system:
   ```bash
   source venv/bin/activate
   python3 main.py train_v3.csv MF 50
   ```
4. Check the output file: `train_v3_recommendations.txt`
   - Each line: `user_id recommended_item_1 recommended_item_2 ... recommended_item_50`

## Troubleshooting

- **Module not found**: Make sure virtual environment is activated and dependencies installed:
  ```bash
  source venv/bin/activate
  pip install -r requirements.txt
  ```
- **CSV format error**: Verify your CSV has columns: `user_id`, `item_id`, `rating`
- **File not found**: Check that the CSV file path is correct

## Model Parameters

You can modify model parameters in `main.py`. Default values:

- **k**: Number of latent dimensions (default: 10)
- **learning_rate**: Learning rate for optimization (default: 0.001)
- **lambda_u**: Regularization for user factors (default: 0.01)
- **lambda_i**: Regularization for item factors (default: 0.01)
- **max_iter**: Number of training iterations (default: 100)

Adjust these values based on your data size and hardware:
- **Larger datasets**: Increase `max_iter` (200-500)
- **More personalization**: Decrease `learning_rate` (0.0001-0.0005)
- **Better accuracy**: Increase `k` (15-20)
