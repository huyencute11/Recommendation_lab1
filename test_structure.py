"""
Test to check exact structure of uid_map and iid_map
"""
from data_loader import RecommendationDataLoader

loader = RecommendationDataLoader('train_v3.csv')
loader.load_data()
dataset = loader.create_dataset()

print(f"uid_map type: {type(dataset.uid_map)}")
print(f"uid_map length: {len(dataset.uid_map)}")
print(f"uid_map[0]: {dataset.uid_map[0]}")
print(f"uid_map[1]: {dataset.uid_map[1]}")
print(f"uid_map[:5]: {dataset.uid_map[:5]}")

print(f"\niid_map type: {type(dataset.iid_map)}")
print(f"iid_map[:5]: {dataset.iid_map[:5]}")
