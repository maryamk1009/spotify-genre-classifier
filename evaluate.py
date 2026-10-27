import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

with open("data/label_map.json") as f:
    label_map = json.load(f)
GENRES = [g for g, i in sorted(label_map.items(), key=lambda kv: kv[1])]
FEATURES = ["duration_ms", "explicit", "danceability", "energy", "key",
            "loudness", "mode", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo",
            "time_signature"]

device = "mps" if torch.backends.mps.is_available() else "cpu"

class GenreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(14, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 8),
        )
    def forward(self, x):
        return self.net(x)

model = GenreNet().to(device)
model.load_state_dict(torch.load("genre_model_v2.pt", map_location=device))
model.eval()

# final test score
d = np.load("data/prepared.npz")
X_test = torch.tensor(d["X_test"], dtype=torch.float32)
y_test = d["y_test"]
with torch.no_grad():
    preds = model(X_test.to(device)).argmax(dim=1).cpu().numpy()
print(f"FINAL test accuracy: {accuracy_score(y_test, preds):.3f}  (random forest test was 0.741)")
print("\nPer-genre report:")
print(classification_report(y_test, preds, target_names=GENRES))

cm = confusion_matrix(y_test, preds)
ConfusionMatrixDisplay(cm, display_labels=GENRES).plot(xticks_rotation=45)
plt.tight_layout(); plt.savefig("confusion_matrix.png")
print("Saved confusion_matrix.png")

# checking w/ zack tabudlo
df = pd.read_csv("data/dataset.csv")
df = df[df["track_genre"].isin(GENRES)].dropna(subset=["artists", "track_name"])
df = df.drop_duplicates(subset="track_id", keep="first")
df["explicit"] = df["explicit"].astype(int)
y_all = df["track_genre"].map(label_map).to_numpy()
idx = np.arange(len(df))
_, idx_temp, _, _ = train_test_split(idx, y_all, test_size=0.30, random_state=42, stratify=y_all)
_, idx_test, _, _ = train_test_split(idx_temp, y_all[idx_temp], test_size=0.50, random_state=42, stratify=y_all[idx_temp])
scaler = StandardScaler().fit(df[FEATURES].to_numpy(dtype=np.float32)[train_test_split(idx, y_all, test_size=0.30, random_state=42, stratify=y_all)[0]])

test_df = df.iloc[idx_test]
zack = test_df[test_df["artists"].str.contains("Zack Tabudlo", na=False)]
print(f"\nZack Tabudlo songs that landed in the test set: {len(zack)}")
if len(zack):
    Xz = torch.tensor(scaler.transform(zack[FEATURES].to_numpy(dtype=np.float32)), dtype=torch.float32)
    with torch.no_grad():
        zp = model(Xz.to(device)).argmax(dim=1).cpu().numpy()
    for (_, row), p in zip(zack.iterrows(), zp):
        print(f"  {row['track_name']}  actual: {row['track_genre']}  predicted: {GENRES[p]}")
