"""
Auto pipeline:
1) Benchmark BPR vs WMF on offline split (F1@K)
2) Select winner
3) Train winner on full implicit data
4) Export submission file with exactly one row per user_id in ascending order
"""
import argparse
from collections import defaultdict
import math

from cornac.models import BPR, WMF

from benchmark_cornac_models import (
    compute_metrics,
    load_positive_df,
    split_train_valid,
    build_seen,
    recommend_all,
    sample_users,
    create_wmf_model,
)
from enhanced_data_loader import EnhancedDataLoader
from cornac.data import Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train best model (BPR/WMF) and export submission.")
    parser.add_argument("--data", type=str, default="train_v3.csv")
    parser.add_argument("--k", type=int, default=50, help="Top-K recommendations per user")
    parser.add_argument("--rating-threshold", type=float, default=3.5)
    parser.add_argument("--holdout-per-user", type=int, default=1)
    parser.add_argument("--min-pos-per-user", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Used by BPR only")
    parser.add_argument("--lambda-reg", type=float, default=0.001)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--benchmark-users",
        type=int,
        default=5000,
        help="Number of users sampled for benchmark scoring (0 = all users)",
    )
    parser.add_argument("--force-model", choices=["BPR", "WMF"], default=None)
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument(
        "--popularity-boost",
        type=float,
        default=0.0,
        help="Popularity prior weight added to model score (default: 0.0 for better personalization)",
    )
    parser.add_argument(
        "--cold-user-min-history",
        type=int,
        default=2,
        help="Users with fewer interactions than this are treated as cold users",
    )
    parser.add_argument(
        "--popularity-penalty",
        type=float,
        default=0.0,
        help="Subtract alpha*log1p(popularity) from score to improve personalization",
    )
    parser.add_argument(
        "--item-index-base",
        type=int,
        default=1,
        choices=[0, 1],
        help="Output item ids as 1-based (default) or 0-based",
    )
    return parser.parse_args()


def evaluate_candidate(model_name, args):
    pos_df = load_positive_df(args.data, args.rating_threshold)
    train_rows, truth, users_eval = split_train_valid(
        pos_df,
        holdout_per_user=args.holdout_per_user,
        min_pos_per_user=args.min_pos_per_user,
    )
    users_eval, truth = sample_users(users_eval, truth, args.benchmark_users, args.seed)
    dataset = Dataset.from_uir(train_rows)
    if model_name == "BPR":
        model = BPR(
            k=args.latent_dim,
            learning_rate=args.learning_rate,
            lambda_reg=args.lambda_reg,
            max_iter=args.max_iter,
            seed=args.seed,
            verbose=False,
        )
    else:
        model = create_wmf_model(
            latent_dim=args.latent_dim,
            lambda_reg=args.lambda_reg,
            max_iter=args.max_iter,
            seed=args.seed,
            verbose=False,
        )

    model.fit(dataset)
    seen = build_seen(train_rows)
    preds = recommend_all(model, dataset, users_eval, seen, args.k)
    return compute_metrics(preds, truth, args.k)


def train_full_and_export(model_name, args):
    loader = EnhancedDataLoader(
        args.data, rating_threshold=args.rating_threshold, ensure_all_users=True
    )
    raw_df = loader.load_data()
    dataset = loader.create_dataset_implicit()
    if dataset is None:
        raise RuntimeError("Failed to create implicit dataset from full data.")

    if model_name == "BPR":
        model = BPR(
            k=args.latent_dim,
            learning_rate=args.learning_rate,
            lambda_reg=args.lambda_reg,
            max_iter=args.max_iter,
            seed=args.seed,
            verbose=True,
        )
    else:
        model = create_wmf_model(
            latent_dim=args.latent_dim,
            lambda_reg=args.lambda_reg,
            max_iter=args.max_iter,
            seed=args.seed,
            verbose=True,
        )
    model.fit(dataset)

    user_ids = dataset.user_ids
    item_ids = dataset.item_ids
    num_users = dataset.num_users
    num_items = dataset.num_items

    user_seen = defaultdict(set)
    for _, row in raw_df.iterrows():
        user_seen[int(row["user_id"])].add(int(row["item_id"]))

    implicit_df = loader.convert_to_implicit_feedback()
    item_popularity = implicit_df["item_id"].value_counts().to_dict()
    popular_items = [int(it) for it, _ in sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)]

    recommendations = {}
    fallback_users = 0
    for user_iid in range(num_users):
        user_id = int(user_ids[user_iid])
        seen_items = user_seen.get(user_id, set())
        user_history_size = len(seen_items)
        scores = []
        for item_iid in range(num_items):
            try:
                item_id = int(item_ids[item_iid])
                if item_id in seen_items:
                    continue
                s = model.score(user_iid, item_iid)
                if args.popularity_boost > 0:
                    s += args.popularity_boost * item_popularity.get(str(item_id), 0)
                if args.popularity_penalty > 0:
                    s -= args.popularity_penalty * math.log1p(item_popularity.get(str(item_id), 0))
                scores.append((item_id, s))
            except Exception:
                continue
        scores.sort(key=lambda x: x[1], reverse=True)
        rec = [str(item_id) for item_id, _ in scores[: args.k]]

        # Fallback only when the user is cold or recommendations are insufficient.
        if len(rec) < args.k:
            fallback_users += 1
            for popular_item in popular_items:
                if popular_item in seen_items:
                    continue
                if str(popular_item) in rec:
                    continue
                rec.append(str(popular_item))
                if len(rec) == args.k:
                    break
        elif user_history_size < args.cold_user_min_history:
            # For very cold users, keep only a short personalized head and fill remainder by popularity.
            keep_n = max(10, args.k // 5)
            rec = rec[:keep_n]
            fallback_users += 1
            for popular_item in popular_items:
                if popular_item in seen_items:
                    continue
                if str(popular_item) in rec:
                    continue
                rec.append(str(popular_item))
                if len(rec) == args.k:
                    break
        recommendations[user_id] = rec

    min_user_id = int(raw_df["user_id"].astype(int).min())
    max_user_id = int(raw_df["user_id"].astype(int).max())
    out_file = args.output or args.data.replace(".csv", "_recommendations.txt")
    row_signatures = defaultdict(int)
    item_frequency = defaultdict(int)

    with open(out_file, "w") as f:
        for user_id in range(min_user_id, max_user_id + 1):
            items = recommendations.get(user_id)
            if not items:
                seen_items = user_seen.get(user_id, set())
                items = []
                fallback_users += 1
                for popular_item in popular_items:
                    if popular_item in seen_items:
                        continue
                    items.append(str(popular_item))
                    if len(items) == args.k:
                        break
            row_signatures[tuple(items)] += 1
            for item in items:
                item_frequency[item] += 1
            if args.item_index_base == 0:
                export_items = [str(int(x) - 1) for x in items]
            else:
                export_items = items
            f.write(" ".join(export_items) + "\n")

    unique_rows = len(row_signatures)
    max_repeat = max(row_signatures.values()) if row_signatures else 0
    top_item_coverage = (max(item_frequency.values()) / (max_user_id - min_user_id + 1)) if item_frequency else 0.0

    print(f"\nSaved submission file: {out_file}")
    print(f"Rows: {max_user_id - min_user_id + 1} | Expected users: {max_user_id}")
    print(f"Top-K per row: {args.k}")
    print(f"Item index base: {args.item_index_base}")
    print(f"Fallback users: {fallback_users}")
    print(f"Unique rows: {unique_rows} | Most repeated row: {max_repeat}")
    print(f"Top-item user coverage: {top_item_coverage:.4f}")


def main():
    args = parse_args()

    print("=" * 72)
    print("Step 1/2: Benchmark model candidates")
    print("=" * 72)

    if args.force_model:
        selected_model = args.force_model
        print(f"Force model enabled: {selected_model}")
    else:
        bpr_p, bpr_r, bpr_f1 = evaluate_candidate("BPR", args)
        print(f"BPR -> P@{args.k}: {bpr_p:.6f} | R@{args.k}: {bpr_r:.6f} | F1@{args.k}: {bpr_f1:.6f}")
        try:
            wmf_p, wmf_r, wmf_f1 = evaluate_candidate("WMF", args)
            print(f"WMF -> P@{args.k}: {wmf_p:.6f} | R@{args.k}: {wmf_r:.6f} | F1@{args.k}: {wmf_f1:.6f}")
        except Exception as exc:
            wmf_p, wmf_r, wmf_f1 = 0.0, 0.0, -1.0
            print(f"WMF -> skipped due to error: {exc}")
        selected_model = "BPR" if bpr_f1 >= wmf_f1 else "WMF"
        print(f"Selected model: {selected_model}")

    print("\n" + "=" * 72)
    print("Step 2/2: Train selected model on full data and export")
    print("=" * 72)
    train_full_and_export(selected_model, args)


if __name__ == "__main__":
    main()
