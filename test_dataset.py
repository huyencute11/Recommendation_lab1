"""
Test script to understand cornac Dataset structure
"""
from data_loader import RecommendationDataLoader

loader = RecommendationDataLoader('train_v3.csv')
loader.load_data()
dataset = loader.create_dataset()

print("Dataset attributes:")
print(dir(dataset))
print("\n")

# Try to get user and item indices
if hasattr(dataset, 'user_indices'):
    print(f"user_indices type: {type(dataset.user_indices)}")
    print(f"First 5 user_indices: {list(dataset.user_indices.items())[:5]}")
else:
    print("No user_indices attribute")

print("\n")

if hasattr(dataset, 'item_indices'):
    print(f"item_indices type: {type(dataset.item_indices)}")
    print(f"First 5 item_indices: {list(dataset.item_indices.items())[:5]}")
else:
    print("No item_indices attribute")

print(f"\nnum_users: {dataset.num_users}")
print(f"num_items: {dataset.num_items}")
