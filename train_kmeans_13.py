"""Train K-Means on Steam games with K=13. Place games.csv beside this script.

Install the dependencies if needed.
    python -m pip install pandas numpy scikit-learn joblib
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import ast
import json
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer, Normalizer

FILE = Path(__file__).resolve().with_name("games.csv")
# Example Windows path
# FILE = Path(r"C:\Users\Computer\Desktop\DataBase\games.csv")
OUTPUT_DIR = FILE.parent / "kmeans_outputs"
K = 13
MIN_TOTAL_REVIEWS = 20
CONTENT_COLUMNS = ["genres", "categories", "tags"]


def column_key(value):
    key = re.sub(r'[^a-z0-9]+', '_', str(value).strip().casefold()).strip('_')
    aliases = {
        'appid': 'app_id', 'steam_appid': 'app_id', 'title': 'name',
        'positive_reviews': 'positive', 'positive_ratings': 'positive',
        'negative_reviews': 'negative', 'negative_ratings': 'negative',
        'average_playtime': 'average_playtime_forever',
        'median_playtime': 'median_playtime_forever',
    }
    return aliases.get(key, key)


def whole_counts(series):
    numbers = pd.to_numeric(series, errors="raise")
    valid = numbers.notna() & np.isfinite(numbers) & (numbers >= 0) & (numbers % 1 == 0)
    if not valid.all():
        raise ValueError(f"Column {series.name} must contain nonnegative whole numbers.")
    return numbers.astype("int64")


def parse_labels(value):
    if value is None:
        return []
    if not isinstance(value, (list, tuple, set, dict)) and pd.isna(value):
        return []

    if isinstance(value, str):
        value = value.strip()
        if value.casefold() in {'', 'null', 'none', 'nan', '<na>'}:
            return []
        for _ in range(2):
            if not isinstance(value, str) or not value or value[0] not in '[{\"\'':
                break
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError, TypeError, RecursionError) as exc:
                    raise ValueError('Invalid list or dictionary format.') from exc
            if isinstance(value, str):
                value = value.strip()

    if value is None:
        return []
    if isinstance(value, str):
        value = re.split(r'[,;|]', value)
    elif isinstance(value, dict):
        if any(k in value for k in ('description', 'name', 'tag')):
            value = [value]
        else:
            value = list(value.keys())
    if not isinstance(value, (list, tuple, set)):
        raise ValueError('Expected a list of labels or a tag dictionary.')

    labels = set()
    for item in value:
        if isinstance(item, dict):
            item = item.get('description', item.get('name', item.get('tag')))
        if not isinstance(item, str):
            raise ValueError('Each content label must be a string.')
        item = unicodedata.normalize('NFKC', item)
        item = re.sub(r'\s+', ' ', item).strip().casefold()
        if item:
            labels.add(item)
    return sorted(labels)


def make_tokens(table):
    profiles = []
    for position, row in enumerate(table[CONTENT_COLUMNS].itertuples(index=False, name=None)):
        tokens = []
        for column, value in zip(CONTENT_COLUMNS, row):
            try:
                tokens.extend(f'{column}:{label}' for label in parse_labels(value))
            except ValueError as exc:
                app_id = table.iloc[position]['app_id']
                raise ValueError(f'Invalid {column} value for app_id={app_id}. {exc}') from exc
        profiles.append(sorted(set(tokens)))
    return profiles


def main():
    # Load the games and apply the minimum review count.
    games = pd.read_csv(FILE, encoding="utf-8-sig", low_memory=False)
    input_rows = len(games)
    column_renames = {col: column_key(col) for col in games.columns}
    games = games.rename(columns=column_renames)
    if games.columns.duplicated().any():
        raise ValueError("Two columns have the same name after normalization.")
    missing = [c for c in ["app_id", "name", *CONTENT_COLUMNS] if c not in games]
    if missing:
        raise ValueError(f"Missing required columns {missing}.")

    have_votes = {"positive", "negative"}.issubset(games.columns)
    if have_votes:
        games["positive"] = whole_counts(games["positive"])
        games["negative"] = whole_counts(games["negative"])
        games["total_reviews"] = games["positive"] + games["negative"]
    elif "total_reviews" in games:
        games["total_reviews"] = whole_counts(games["total_reviews"])
    else:
        raise ValueError("The review filter requires total_reviews or both positive and negative.")

    meets_review_threshold = games["total_reviews"] >= MIN_TOTAL_REVIEWS
    has_name = games["name"].fillna("").astype(str).str.strip().ne("")
    review_eligible_rows = int(meets_review_threshold.sum())
    skipped_missing_names = int((meets_review_threshold & ~has_name).sum())
    games = games.loc[meets_review_threshold & has_name].copy().reset_index(drop=True)
    if have_votes:
        games["rating_ratio"] = games["positive"] / games["total_reviews"]
    games["app_id"] = whole_counts(games["app_id"])
    if (games["app_id"] == 0).any() or games["app_id"].duplicated().any():
        raise ValueError("Each game must have a unique, positive app_id.")

    print(f"Loaded {input_rows:,} games.")
    print(f"{review_eligible_rows:,} games have at least {MIN_TOTAL_REVIEWS} reviews.")
    print(f"Skipped {skipped_missing_names:,} review-eligible games with a missing or blank name.")

    # Encode the content features and fit K-Means.
    profiles = make_tokens(games)
    positions = [i for i, tokens in enumerate(profiles) if tokens]
    training_profiles = [profiles[i] for i in positions]
    distinct_profiles = len({tuple(tokens) for tokens in training_profiles})
    if distinct_profiles < K:
        raise ValueError(f"K={K} requires at least {K} distinct content profiles. Found {distinct_profiles}.")

    encoder = MultiLabelBinarizer(sparse_output=True)
    encoded = encoder.fit_transform(training_profiles).astype(np.float32)
    normalizer = Normalizer(norm="l2")
    X = normalizer.fit_transform(encoded)

    model = KMeans(n_clusters=K, n_init=10, max_iter=300, random_state=42, algorithm="lloyd")
    labels = model.fit_predict(X)
    if np.unique(labels).size != K:
        raise RuntimeError("K-Means assigned games to fewer than 13 clusters. Check the variety of game content.")

    # Match each cluster ID to the game row used for training.
    assigned = pd.Series(labels, index=games.index[positions], dtype="Int64")
    games["kmeans_cluster"] = assigned.reindex(games.index)
    games["kmeans_status"] = np.where(games["kmeans_cluster"].notna(), "clustered", "missing_content")
    print(f"Assigned {len(labels):,} games to {model.n_clusters} clusters.")
    print(games["kmeans_cluster"].value_counts().sort_index().rename("game_count"))

    # Save the games table for KNN and the fitted model.
    summary = games.dropna(subset=["kmeans_cluster"]).groupby("kmeans_cluster").agg(
        game_count=("app_id", "size"),
        example_games=("name", lambda names: " | ".join(names.iloc[:5].astype(str))),
    )
    top_features = []
    for center in model.cluster_centers_:
        top = np.argsort(center)[::-1]
        top = top[center[top] > 0][:10]
        top_features.append(" | ".join(encoder.classes_[top]))
    summary["top_features"] = top_features
    summary = summary[["game_count", "top_features", "example_games"]]

    metadata = {
        "project": "Smart Game Recommendation System", "stage": "K-Means", "schema_version": 1,
        "selected_k": K, "selection_method": "fixed", "manual_override": True,
        "random_seed": model.random_state, "final_n_init": model.n_init,
        "content_columns": CONTENT_COLUMNS, "feature_count": X.shape[1],
        "feature_encoding": "column-prefixed binary labels, then row L2 normalization",
        "cluster_column": "kmeans_cluster", "status_column": "kmeans_status",
        "minimum_total_reviews": MIN_TOTAL_REVIEWS, "input_file": FILE.name,
        "rows_input": input_rows, "rows_total": len(games), "rows_clustered": len(labels),
        "rows_below_review_threshold": input_rows - review_eligible_rows,
        "rows_skipped_missing_name": skipped_missing_names,
        "rows_missing_content": len(games) - len(labels),
        "column_renames": column_renames, "final_inertia": float(model.inertia_),
        "versions": {"scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    bundle = {"model": model, "encoder": encoder, "normalizer": normalizer, "metadata": metadata}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUTPUT_DIR) as temp_dir:
        stage = Path(temp_dir)
        games.to_csv(stage / "games.csv", index=False, encoding="utf-8-sig")
        joblib.dump(bundle, stage / "games_kmeans.joblib", compress=3)
        summary.to_csv(stage / "cluster_summary.csv", encoding="utf-8-sig")
        (stage / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        for output in stage.iterdir():
            output.replace(OUTPUT_DIR / output.name)

    print(f"Saved the games table for KNN to {(OUTPUT_DIR / 'games.csv').resolve()}")


if __name__ == "__main__":
    main()
