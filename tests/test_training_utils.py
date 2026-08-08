from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.training.checkpointing import load_checkpoint, save_checkpoint
from src.training.early_stopping import EarlyStopping
from src.training.engine import evaluate, train_one_epoch


class TinyDataset(Dataset):
    def __init__(self) -> None:
        self.x = torch.tensor(
            [[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]]
        )
        self.y = torch.tensor([0, 0, 1, 1])

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict:
        return {"x": self.x[index], "y": self.y[index]}


def test_engine_trains_and_evaluates_toy_classifier() -> None:
    torch.manual_seed(42)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TinyDataset(), batch_size=4, shuffle=False)

    first = train_one_epoch(
        model,
        loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
    )
    for _ in range(20):
        train_one_epoch(
            model,
            loader,
            optimizer=optimizer,
            criterion=criterion,
            device=torch.device("cpu"),
        )
    metrics = evaluate(
        model, loader, criterion=criterion, device=torch.device("cpu")
    )

    assert first["mean_grad_norm"] > 0
    assert metrics["accuracy"] == 1.0


def test_checkpoint_round_trip_restores_identical_logits(tmp_path: Path) -> None:
    torch.manual_seed(42)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.randn(4, 3)
    expected = model(inputs).detach().clone()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        {
            "epoch": 3,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)

    payload = load_checkpoint(path, model=model, optimizer=optimizer)

    assert payload["epoch"] == 3
    assert torch.equal(model(inputs), expected)

    labels = torch.tensor([0, 1, 0, 1])
    optimizer.zero_grad(set_to_none=True)
    nn.CrossEntropyLoss()(model(inputs), labels).backward()
    optimizer.step()
    assert not torch.equal(model(inputs), expected)


def test_early_stopping_uses_patience() -> None:
    state = EarlyStopping(patience=2)

    assert state.update(0.5) == (True, False)
    assert state.update(0.4) == (False, False)
    assert state.update(0.4) == (False, True)
