"""
Benchmark Cornac implicit models (BPR, WMF) using offline F1@K.
"""
import argparse
from collections import defaultdict
import random

import pandas as pd
from cornac.data import Dataset
from cornac.models import BPR, WMF


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark BPR and WMF on offline split.")
    parser.add_argument("--data", type=str, default="train_v3.csv")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--rating-threshold", type=float, default=3.5)
    parser.add_argument("--holdout-per-user", type=int, default=1)
    parser.add_argument("--min-pos-per-user", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--lambda-reg", type=float, default=0.001)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--benchmark-users",
        type=int,
        default=5000,
        help="Number of users sampled for benchmark scoring (0 = all users)",
    )
    return parser.parse_args()


def load_positive_df(path, threshold):
    df = pd.read_csv(path)
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    pos = df[df["rating"] >= threshold].copy()
    pos = pos.drop_duplicates(subset=["user_id", "item_id"])
    return pos


def split_train_valid(pos_df, holdout_per_user=1, min_pos_per_user=2):
    user_items = defaultdict(list)
    for _, row in pos_df.iterrows():
        user_items[int(row["user_id"])].append(int(row["item_id"]))

    train_rows = []
    truth = {}
    users_eval = sorted(user_items.keys())
    for user_id in users_eval:
        items = sorted(set(user_items[user_id]))
        if len(items) >= min_pos_per_user:
            n_holdout = min(holdout_per_user, len(items) - 1)
            valid_items = items[-n_holdout:]
            train_items = items[:-n_holdout]
            truth[user_id] = set(valid_items)
        else:
            train_items = items

        for item_id in train_items:
            train_rows.append((str(user_id), str(item_id), 1.0))

    return train_rows, truth, users_eval


def sample_users(users_eval, truth, benchmark_users, seed):
    if benchmark_users <= 0 or benchmark_users >= len(users_eval):
        return users_eval, truth
    rnd = random.Random(seed)
    sampled_users = set(rnd.sample(users_eval, benchmark_users))
    sampled_truth = {u: rel for u, rel in truth.items() if u in sampled_users}
    return sorted(sampled_users), sampled_truth


def build_seen(train_rows):
    seen = defaultdict(set)
    for u, i, _ in train_rows:
        seen[int(u)].add(int(i))
    return seen


def recommend_all(model, dataset, users_eval, seen, topk):
    preds = {}
    item_ids = dataset.item_ids
    num_items = dataset.num_items

    for user_id in users_eval:
        uid = str(user_id)
        if uid not in dataset.uid_map:
            preds[user_id] = []
            continue
        user_iid = dataset.uid_map[uid]
        user_seen = seen.get(user_id, set())
        scores = []
        for item_iid in range(num_items):
            try:
                item_id = int(item_ids[item_iid])
                if item_id in user_seen:
                    continue
                scores.append((item_id, model.score(user_iid, item_iid)))
            except Exception:
                continue
        scores.sort(key=lambda x: x[1], reverse=True)
        preds[user_id] = [i for i, _ in scores[:topk]]
    return preds


def compute_metrics(preds, truth, k):
    ps, rs, f1s = [], [], []
    for user_id, rel_items in truth.items():
        rec_items = preds.get(user_id, [])[:k]
        hit = len(set(rec_items) & rel_items)
        p = hit / k
        r = hit / len(rel_items)
        f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
    if not f1s:
        return 0.0, 0.0, 0.0
    return sum(ps) / len(ps), sum(rs) / len(rs), sum(f1s) / len(f1s)


def evaluate_model(name, model, train_rows, truth, users_eval, topk):
    dataset = Dataset.from_uir(train_rows)
    model.fit(dataset)
    seen = build_seen(train_rows)
    preds = recommend_all(model, dataset, users_eval, seen, topk)
    return compute_metrics(preds, truth, topk)


def create_wmf_model(latent_dim, lambda_reg, max_iter, seed, verbose=False):
    # Cornac versions differ in WMF constructor; try safest signatures.
    try:
        return WMF(
            k=latent_dim,
            lambda_reg=lambda_reg,
            max_iter=max_iter,
            seed=seed,
            verbose=verbose,
        )
    except TypeError:
        try:
            return WMF(
                k=latent_dim,
                lambda_reg=lambda_reg,
                max_iter=max_iter,
                verbose=verbose,
            )
        except TypeError:
            return WMF(
                k=latent_dim,
                max_iter=max_iter,
                verbose=verbose,
            )


def main():
    args = parse_args()
    pos_df = load_positive_df(args.data, args.rating_threshold)
    train_rows, truth, users_eval = split_train_valid(
        pos_df, holdout_per_user=args.holdout_per_user, min_pos_per_user=args.min_pos_per_user
    )
    users_eval, truth = sample_users(users_eval, truth, args.benchmark_users, args.seed)

    print("=" * 72)
    print("Benchmark Cornac Models (Implicit)")
    print("=" * 72)
    print(f"Train rows: {len(train_rows)} | Eval users: {len(users_eval)} | Truth users: {len(truth)}")

    bpr = BPR(
        k=args.latent_dim,
        learning_rate=args.learning_rate,
        lambda_reg=args.lambda_reg,
        max_iter=args.max_iter,
        seed=args.seed,
        verbose=False,
    )
    bpr_p, bpr_r, bpr_f1 = evaluate_model("BPR", bpr, train_rows, truth, users_eval, args.k)
    print(f"BPR -> P@{args.k}: {bpr_p:.6f} | R@{args.k}: {bpr_r:.6f} | F1@{args.k}: {bpr_f1:.6f}")

    try:
        wmf = create_wmf_model(
            latent_dim=args.latent_dim,
            lambda_reg=args.lambda_reg,
            max_iter=args.max_iter,
            seed=args.seed,
            verbose=False,
        )
        wmf_p, wmf_r, wmf_f1 = evaluate_model("WMF", wmf, train_rows, truth, users_eval, args.k)
        print(f"WMF -> P@{args.k}: {wmf_p:.6f} | R@{args.k}: {wmf_r:.6f} | F1@{args.k}: {wmf_f1:.6f}")
    except Exception as exc:
        wmf_p, wmf_r, wmf_f1 = 0.0, 0.0, -1.0
        print(f"WMF -> skipped due to error: {exc}")

    winner = "BPR" if bpr_f1 >= wmf_f1 else "WMF"
    print("-" * 72)
    print(f"Best model by F1@{args.k}: {winner}")
    print("=" * 72)


if __name__ == "__main__":
    main()
