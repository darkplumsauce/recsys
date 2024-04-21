import torch
from torch.utils.data import random_split, DataLoader

from data import MovieLensData
from model import AutoRec
from utils import load_config

cfg = load_config("./config/config.yml")

dataset = MovieLensData(mode=cfg["mode"])

# Split data
train_data, val_data, test_data = random_split(dataset, cfg["data_params"]["split_frac"])

# Batch and shuffle data
train_dataloader = DataLoader(train_data, batch_size=cfg["data_params"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=cfg["data_params"]["batch_size"], shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=cfg["data_params"]["batch_size"], shuffle=False)

model = AutoRec(dataset.get_len_vec(), cfg["model_params"]["hidden_units"])
optimizer = torch.optim.Rprop(model.parameters())  # NOTE: use RProp optimizer
loss_func = torch.nn.MSELoss()  # NOTE: use MSE loss

for epoch in range(cfg["model_params"]["epochs"]):
    print("Epoch", epoch)
    # Train
    model.train()
    train_loss = list()
    for batch in train_dataloader:
        optimizer.zero_grad()

        pred = model(batch)

        loss = loss_func(pred, batch)  # auto-encoder means to recover original input, i.e. target = original
        train_loss.append(loss)

        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    val_loss = list()
    for batch in val_dataloader:
        batch[batch == 0.0] = 3.0  # NOTE: assign 3.0 default value to unobserved ratings (incl. val)
        with torch.no_grad():
            pred = model(batch)
        loss = loss_func(pred, batch)
        val_loss.append(loss)

    print("Train loss (MSE):", sum(train_loss) / len(train_loss))
    print("Val loss (MSE):", sum(val_loss) / len(val_loss))

# Test
model.eval()
test_error = list()
for batch in test_dataloader:
    batch[batch == 0.0] = 3.0  # NOTE: assign 3.0 default value to unobserved ratings (incl. test)
    with torch.no_grad():
        pred = model(batch)
    error = torch.sqrt(loss_func(pred, batch))  # NOTE: use RMSE error
    test_error.append(error)
print("Test error (RMSE)", sum(test_error) / len(test_error))
