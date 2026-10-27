import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# use my Mac's GPU cores if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Training on:", device)

# load the prepped data, turn into pytorch tensors
d = np.load("data/prepared.npz")
X_train = torch.tensor(d["X_train"], dtype=torch.float32)
y_train = torch.tensor(d["y_train"], dtype=torch.long)
X_val = torch.tensor(d["X_val"], dtype=torch.float32)
y_val = torch.tensor(d["y_val"], dtype=torch.long)

train_ds = TensorDataset(X_train, y_train)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

# the model
class GenreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(14, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 8),
        )
    def forward(self, x):
        return self.net(x)

model = GenreNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# the training loop
train_losses, val_accs = [], []
for epoch in range(30):
    model.train()
    total_loss = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()        # clear last round's gradients
        preds = model(xb)            # forward: make guesses
        loss = loss_fn(preds, yb)    # how wrong were we?
        loss.backward()              # backprop: blame each weight
        optimizer.step()             # nudge weights to be less wrong
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_dl))

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val.to(device)).argmax(dim=1).cpu()
    acc = (val_preds == y_val).float().mean().item()
    val_accs.append(acc)
    print(f"Epoch {epoch+1:2d}  loss {train_losses[-1]:.4f}  val acc {acc:.3f}")

print(f"\nBest val accuracy: {max(val_accs):.3f}  (random forest baseline: 0.744)")

# save proof
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(train_losses); ax[0].set_title("Training loss per epoch")
ax[1].plot(val_accs); ax[1].set_title("Validation accuracy per epoch")
plt.tight_layout(); plt.savefig("training_curves.png")
torch.save(model.state_dict(), "genre_model.pt")
print("Saved training_curves.png and genre_model.pt")
