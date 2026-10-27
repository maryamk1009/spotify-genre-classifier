import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

GENRES = ["acoustic", "classical", "jazz", "hip-hop",
        "heavy-metal", "edm", "country", "reggae"]

FEATURES = ["duration_ms", "explicit", "danceability", "energy", "key",
            "loudness", "mode", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo",
            "time_signature"]

# Load, keep only our 8 genres, drop broken row
df = pd.read_csv("data/dataset.csv")
df = df[df["track_genre"].isin(GENRES)]
df = df.dropna(subset=["artists", "track_name"])
print("After genre filter:", len(df), "rows")

# Dedupe by track_id
before = len(df)
df = df.drop_duplicates(subset="track_id", keep="first")
print(f"Dedupe removed {before - len(df)} duplicate rows -> {len(df)} rows left")
print("\nRows per genre after dedupe:")
print(df["track_genre"].value_counts())

# explicit is True/False text
df["explicit"] = df["explicit"].astype(int)

#  Turn genre names into numbers 0-7
label_map = {genre: i for i, genre in enumerate(GENRES)}
df["label"] = df["track_genre"].map(label_map)
with open("data/label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

X = df[FEATURES].to_numpy(dtype=np.float32)
y = df["label"].to_numpy(dtype=np.int64)

# Split 70% train / 15% validation / 15% test, stratified
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
print(f"\nTrain: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# Scale features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Save everything for the model scripts
np.savez("data/prepared.npz",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test)
print("Saved data/prepared.npz")
