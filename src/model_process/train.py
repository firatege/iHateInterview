("""Train a classifier on blendshape data.

Reads a CSV with blendshape columns plus an `emotion` column.
Performs train/test split, scaling, simple hyperparameter search for
RandomForest and LogisticRegression, selects the best model by
cross-validated score, saves the model and scaler with joblib, and
writes a short report to disk.

Usage:
  python -m src.model_process.train --data data/blendshapes_only.csv --out models/emotion_rf_model.joblib

""")
from pathlib import Path
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import time


def parse_args():
	p = argparse.ArgumentParser(description="Train emotion classifier from blendshape CSV")
	p.add_argument("--data", type=str, default=None, help="Path to the CSV dataset (default tries data/blendshapes_only.csv then data/blendshapes_dataset.csv)")
	p.add_argument("--out", type=str, default="models/emotion_rf_model.joblib", help="Output model path")
	p.add_argument("--scaler-out", type=str, default="models/scaler.joblib", help="Output scaler path")
	p.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
	p.add_argument("--random-state", type=int, default=42)
	p.add_argument("--cv", type=int, default=3, help="Cross-validation folds for search")
	return p.parse_args()


def load_data(path: Path):
	if not path.exists():
		raise FileNotFoundError(f"Data file not found: {path}")
	df = pd.read_csv(path)
	if "emotion" not in df.columns:
		raise ValueError("Dataset must contain an 'emotion' column with labels")
	# Drop rows with NaN
	df = df.dropna()
	X = df.drop(columns=["emotion"]).values.astype(np.float32)
	y = df["emotion"].values.astype(str)
	return X, y


def build_and_search(X_train, y_train, cv=3, random_state=42):
	results = {}

	# RandomForest search (randomized)
	rf = RandomForestClassifier(random_state=random_state)
	rf_param_dist = {
		"n_estimators": [100, 200, 400],
		"max_depth": [None, 10, 20, 40],
		"min_samples_split": [2, 5, 10],
		"min_samples_leaf": [1, 2, 4]
	}
	rs = RandomizedSearchCV(rf, rf_param_dist, n_iter=8, cv=cv, random_state=random_state, n_jobs=-1)
	rs.fit(X_train, y_train)
	results["random_forest"] = rs

	# Logistic Regression grid
	lr = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=random_state)
	lr_param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
	gs = GridSearchCV(lr, lr_param_grid, cv=cv, n_jobs=-1)
	gs.fit(X_train, y_train)
	results["logistic_regression"] = gs

	return results


def pick_best(results):
	best_name = None
	best_estimator = None
	best_score = -1
	for name, res in results.items():
		score = res.best_score_ if hasattr(res, "best_score_") else getattr(res, "score", -1)
		if score > best_score:
			best_score = score
			best_name = name
			best_estimator = res.best_estimator_
	return best_name, best_estimator, best_score


def main():
	args = parse_args()
	data_path = Path(args.data) if args.data else None
	if data_path is None:
		# prefer blendshapes_only.csv as the default dataset
		candidates = [Path("data/blendshapes_only.csv"), Path("data/blendshapes_dataset.csv")]
		data_path = next((p for p in candidates if p.exists()), None)
		if data_path is None:
			raise FileNotFoundError("No dataset found. Place a CSV at data/blendshapes_only.csv or pass --data")

	print(f"Loading data from: {data_path}")
	X, y = load_data(data_path)
	print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")

	le = LabelEncoder()
	y_enc = le.fit_transform(y)

	X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=args.test_size, random_state=args.random_state, stratify=y_enc)

	scaler = StandardScaler()
	X_train_s = scaler.fit_transform(X_train)
	X_test_s = scaler.transform(X_test)

	print("Starting hyperparameter search...")
	t0 = time.time()
	results = build_and_search(X_train_s, y_train, cv=args.cv, random_state=args.random_state)
	t1 = time.time()
	print(f"Search finished in {t1-t0:.1f}s")

	best_name, best_estimator, best_score = pick_best(results)
	if best_estimator is None:
		raise RuntimeError("No best estimator found from search results")
	print(f"Best model: {best_name} (CV score {best_score:.4f})")

	# Final evaluation
	y_pred = best_estimator.predict(X_test_s)
	acc = accuracy_score(y_test, y_pred)
	report = classification_report(y_test, y_pred, target_names=le.classes_)
	print(f"Test accuracy: {acc:.4f}")
	print(report)

	# Save model + scaler + label encoder
	out_model = Path(args.out)
	out_model.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(best_estimator, out_model)
	joblib.dump(scaler, Path(args.scaler_out))
	joblib.dump(le, Path(out_model.parent / "label_encoder.joblib"))

	# Save a short training report
	report_path = out_model.parent / "training_report.txt"
	with open(report_path, "w", encoding="utf-8") as f:
		f.write(f"best_model: {best_name}\n")
		f.write(f"cv_score: {best_score:.4f}\n")
		f.write(f"test_accuracy: {acc:.4f}\n\n")
		f.write(str(report))

	print(f"Model saved to {out_model}")
	print(f"Scaler saved to {args.scaler_out}")
	print(f"Report saved to {report_path}")


if __name__ == "__main__":
	main()

