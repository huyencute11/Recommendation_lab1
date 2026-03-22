"""
Test to check exact structure of uid_map and iid_map
"""
from data_loader import RecommendationDataLoader

loader = RecommendationDataLoader('train_v3.csv')
loader.load_data()
dataset = loader.create_dataset()

print(f"uid_map type: {type(dataset.uid_map)}")
print(f"uid_map (first 5 items): {list(dataset.uid_map.items())[:5]}")

print(f"\niid_map type: {type(dataset.iid_map)}")
print(f"iid_map (first 5 items): {list(dataset.iid_map.items())[:5]}")

print(f"\nuser_ids: {dataset.user_ids[:5]}")
print(f"item_ids: {dataset.item_ids[:5]}")

# So uid_map and iid_map are OrderedDict where:
# key = original_id (string)
# value = internal_index (int)
print("\nTo reverse map (internal_index -> original_id):")
print("Create {value: key for key, value in dataset.uid_map.items()}")
uid_reverse = {v: k for k, v in dataset.uid_map.items()}
print(f"uid_reverse[0] = {uid_reverse[0]}")
print(f"uid_reverse[1] = {uid_reverse[1]}")
