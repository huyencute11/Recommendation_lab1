"""
Test script to understand cornac mapping
"""
from data_loader import RecommendationDataLoader

loader = RecommendationDataLoader('train_v3.csv')
loader.load_data()
dataset = loader.create_dataset()

print("uid_map (first 5):")
print(list(dataset.uid_map.items())[:5])

print("\niid_map (first 5):")
print(list(dataset.iid_map.items())[:5])

print(f"\nuser_ids: {dataset.user_ids[:5]}")
print(f"item_ids: {dataset.item_ids[:5]}")

print("\nTesting user_iter:")
count = 0
for user in dataset.user_iter():
    print(f"User: {user}")
    count += 1
    if count >= 3:
        break

print("\nTesting item_iter:")
count = 0
for item in dataset.item_iter():
    print(f"Item: {item}")
    count += 1
    if count >= 3:
        break
