import pandas as pd
import matplotlib.pyplot as plt

# First, we load the dataset in
df = pd.read_csv("data/dataset.csv")
print("Shape (rows, columns):", df.shape)
print("\nColumns:", list(df.columns))

# Next we can look at the first few songs
print("\nFirst 5 rows:")
print(df[["track_name", "artists", "track_genre"]].head())

# Looking at how many songs per genre, so we can see what genres to keep
print("\nSongs per genre (all of them):")
print(df["track_genre"].value_counts())

# Looking for any missing data
print("\nMissing values per column:")
print(df.isna().sum())

# Numerical values of audio features
features = ["danceability", "energy", "valence", "tempo",
            "acousticness", "instrumentalness", "speechiness",
            "liveness", "loudness", "duration_ms", "popularity"]
print("\nFeature summary:")
print(df[features].describe().round(2))

# Bar chart of genre sizes
counts = df["track_genre"].value_counts()
counts.plot(kind="bar", figsize=(14, 4))
plt.ylabel("Number of tracks")
plt.title("Tracks per genre")
plt.tight_layout()
plt.savefig("genre_counts.png")
print("\nSaved genre_counts.png")

# test case
zack = df[df["artists"].str.contains("Zack Tabudlo", na=False)]
print("\nZack Tabudlo rows found:", len(zack))
print(zack[["track_name", "track_genre", "popularity"]])