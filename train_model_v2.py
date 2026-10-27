import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import copy

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Training on:", device)

d = np.load("data/prepared.npz")
X_train = torch.tensor(d["X_train"], dtype=torch.float32)
y_train = torch.tensor(d["y_train"], dtype=torch.long)
X_val = torch.tensor(d["X_val"], dtype=torch.float32)
y_val = torch.tensor(d["y_val"], dtype=torch.long)

train_dl = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

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
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses, val_accs = [], []
best_acc, best_state = 0.0, None
for epoch in range(80):
    model.train()
    total_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_dl))

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val.to(device)).argmax(dim=1).cpu()
    acc = (val_preds == y_val).float().mean().item()
    val_accs.append(acc)
    if acc > best_acc:
        best_acc = acc
        best_state = copy.deepcopy(model.state_dict())
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:2d}  loss {train_losses[-1]:.4f}  val acc {acc:.3f}")

print(f"\nBest val accuracy: {best_acc:.3f}  (v1: 0.656, random forest: 0.744)")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(train_losses); ax[0].set_title("Training loss per epoch (v2)")
ax[1].plot(val_accs); ax[1].set_title("Validation accuracy per epoch (v2)")
plt.tight_layout(); plt.savefig("training_curves_v2.png")
torch.save(best_state, "genre_model_v2.pt")
print("Saved training_curves_v2.png and genre_model_v2.pt (BEST epoch weights)")
