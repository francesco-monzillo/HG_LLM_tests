import torch
from torch import nn
import HGAT_NN

N = 6
E = 3
F_in = 256
X = torch.randn(N, F_in)
H = torch.tensor([
    [1,0,1],  # node 0 in hyperedge 0 and 2
    [1,1,0],  # node 1 in 0 and 1
    [0,1,0],  # node 2 in 1
    [0,1,1],  # node 3 in 1 and 2
    [1,0,0],  # node 4 in 0
    [0,1,1],  # node 5 in 2
], dtype=torch.float32)

num_classes = 6
labels = torch.randint(0, num_classes, (N,))

model = HGAT_NN.HGAT(in_feats=F_in, hidden_dim=16, out_feats=num_classes, dropout=0.3)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

model.train()
for epoch in range(10000):
    opt.zero_grad()
    logits = model(X, H)   # [N, C]
    loss = loss_fn(logits, labels)
    loss.backward()
    opt.step()
    if epoch % 50 == 0:
        print(f"epoch {epoch} loss {loss.item():.4f}")

# Inference
model.eval()
with torch.no_grad():
    logits = model(X, H)
    preds = logits.argmax(dim=1)
    print("preds:", preds)
    print("labels:", labels)