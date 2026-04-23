import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import glob
import random
from collections import defaultdict
from pathlib import Path

def _get_game_name(path):
    return Path(path).relative_to('latent_actions_dump').parts[0]


def _format_actions(actions, num_samples, file_path):
    actions = torch.as_tensor(actions)
    if actions.ndim == 0:
        actions = actions.unsqueeze(0)
    if actions.shape[0] != num_samples:
        raise ValueError(
            f"Action count mismatch in {file_path}: z_mu has {num_samples} samples but actions has shape {tuple(actions.shape)}"
        )
    return actions


def _build_dataset(samples):
    z = torch.stack([sample[0] for sample in samples], dim=0)
    actions = torch.stack([sample[1] for sample in samples], dim=0)
    games = [sample[2] for sample in samples]
    return z, actions, games


def load_data(test_ratio=0.2, seed=42):
    files = sorted(glob.glob('latent_actions_dump/*/*/*/latent_actions.pt'))
    if not files:
        raise RuntimeError('No latent_actions.pt files found under latent_actions_dump/.')

    samples_by_game = defaultdict(list)
    unique_games = []

    for f in files:
        game_name = _get_game_name(f)
        if game_name not in samples_by_game:
            unique_games.append(game_name)

        data = torch.load(f, map_location='cpu')
        z = torch.as_tensor(data['z_mu'], dtype=torch.float32)
        if z.ndim == 1:
            z = z.unsqueeze(0)

        actions = _format_actions(data['actions'], z.shape[0], f)
        if actions.ndim == 2 and actions.shape[1] == 1 and torch.all(actions == actions.long().to(actions.dtype)):
            actions = actions.squeeze(1)

        for sample_z, sample_action in zip(z, actions):
            samples_by_game[game_name].append((sample_z, sample_action, game_name))

    min_count = min(len(samples) for samples in samples_by_game.values())
    if min_count < 2:
        raise RuntimeError('Need at least two samples per game to create a train/test split.')

    test_per_game = max(1, int(round(min_count * test_ratio)))
    test_per_game = min(test_per_game, min_count - 1)

    rng = random.Random(seed)
    train_samples = []
    test_samples = []
    for game_name in unique_games:
        game_samples = list(samples_by_game[game_name])
        rng.shuffle(game_samples)
        test_samples.extend(game_samples[:test_per_game])
        train_samples.extend(game_samples[test_per_game:])

    train_z, train_actions, train_games = _build_dataset(train_samples)
    test_z, test_actions, test_games = _build_dataset(test_samples)

    if train_actions.ndim == 1:
        train_actions = train_actions.view(-1)
        test_actions = test_actions.view(-1)
        num_actions = int(torch.max(torch.cat([train_actions, test_actions])).item()) + 1
        action_mode = 'multiclass'
    else:
        num_actions = train_actions.shape[1]
        action_mode = 'multilabel'

    game_to_idx = {game_name: idx for idx, game_name in enumerate(unique_games)}
    train_games = torch.tensor([game_to_idx[game_name] for game_name in train_games], dtype=torch.long)
    test_games = torch.tensor([game_to_idx[game_name] for game_name in test_games], dtype=torch.long)

    train_dataset = TensorDataset(train_z, train_actions, train_games)
    test_dataset = TensorDataset(test_z, test_actions, test_games)

    return train_dataset, test_dataset, num_actions, unique_games, action_mode


def _accuracy_from_logits(logits, targets, action_mode):
    if action_mode == 'multiclass':
        predictions = logits.argmax(dim=1)
        return (predictions == targets.long()).float()

    predictions = (torch.sigmoid(logits) >= 0.5).to(targets.dtype)
    return (predictions == targets).view(targets.size(0), -1).all(dim=1).float()


def evaluate(model, loader, action_mode, unique_games):
    model.eval()
    total_correct = 0.0
    total_count = 0
    per_game_correct = {game_name: 0.0 for game_name in unique_games}
    per_game_count = {game_name: 0 for game_name in unique_games}

    with torch.no_grad():
        for batch_z, batch_actions, batch_games in loader:
            logits = model(batch_z)
            batch_correct = _accuracy_from_logits(logits, batch_actions, action_mode)
            total_correct += batch_correct.sum().item()
            total_count += batch_correct.numel()

            for game_idx in batch_games.unique(sorted=True):
                game_mask = batch_games == game_idx
                game_name = unique_games[game_idx.item()]
                per_game_correct[game_name] += batch_correct[game_mask].sum().item()
                per_game_count[game_name] += game_mask.sum().item()

    total_accuracy = total_correct / total_count if total_count else 0.0
    per_game_accuracy = {
        game_name: (per_game_correct[game_name] / per_game_count[game_name] if per_game_count[game_name] else 0.0)
        for game_name in unique_games
    }
    return total_accuracy, per_game_accuracy


def train_multiclass_model(model, loader, criterion, optimizer, epochs, target_index=1):
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_z, batch_targets, batch_games in loader:
            if target_index == 1:
                targets = batch_targets
            elif target_index == 2:
                targets = batch_games
            else:
                raise ValueError(f"Unsupported target_index: {target_index}")

            optimizer.zero_grad()
            logits = model(batch_z)
            loss = criterion(logits, targets.long().view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}")


def evaluate_multiclass_model(model, loader, unique_games, target_index=1):
    model.eval()
    total_correct = 0.0
    total_count = 0
    per_game_correct = {game_name: 0.0 for game_name in unique_games}
    per_game_count = {game_name: 0 for game_name in unique_games}

    with torch.no_grad():
        for batch_z, batch_targets, batch_games in loader:
            if target_index == 1:
                targets = batch_targets
            elif target_index == 2:
                targets = batch_games
            else:
                raise ValueError(f"Unsupported target_index: {target_index}")

            logits = model(batch_z)
            predictions = logits.argmax(dim=1)
            batch_correct = (predictions == targets.long().view(-1)).float()
            total_correct += batch_correct.sum().item()
            total_count += batch_correct.numel()

            for game_idx in batch_games.unique(sorted=True):
                game_mask = batch_games == game_idx
                game_name = unique_games[game_idx.item()]
                per_game_correct[game_name] += batch_correct[game_mask].sum().item()
                per_game_count[game_name] += game_mask.sum().item()

    total_accuracy = total_correct / total_count if total_count else 0.0
    per_game_accuracy = {
        game_name: (per_game_correct[game_name] / per_game_count[game_name] if per_game_count[game_name] else 0.0)
        for game_name in unique_games
    }
    return total_accuracy, per_game_accuracy

def main():
    print("Loading data...")
    train_dataset, test_dataset, num_actions, unique_games, action_mode = load_data()
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Games: {unique_games}")
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # The linear model: predict action from latent encoding
    model = nn.Linear(train_dataset.tensors[0].shape[1], num_actions)
    criterion = nn.CrossEntropyLoss() if action_mode == 'multiclass' else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 100
    print("Training started...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_z, batch_actions, _ in train_loader:
            optimizer.zero_grad()
            pred = model(batch_z)
            if action_mode == 'multiclass':
                loss = criterion(pred, batch_actions.long().view(-1))
            else:
                loss = criterion(pred, batch_actions.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    print("Testing started...")
    total_accuracy, per_game_accuracy = evaluate(model, test_loader, action_mode, unique_games)
    print(f"Total accuracy: {total_accuracy:.4f}")
    for game_name in unique_games:
        print(f"{game_name}: {per_game_accuracy[game_name]:.4f}")

    print("Training game predictor...")
    game_model = nn.Linear(train_dataset.tensors[0].shape[1], len(unique_games))
    game_criterion = nn.CrossEntropyLoss()
    game_optimizer = optim.Adam(game_model.parameters(), lr=1e-3)
    train_multiclass_model(game_model, train_loader, game_criterion, game_optimizer, epochs, target_index=2)

    print("Testing game predictor...")
    game_accuracy, per_game_game_accuracy = evaluate_multiclass_model(game_model, test_loader, unique_games, target_index=2)
    print(f"Game accuracy: {game_accuracy:.4f}")
    for game_name in unique_games:
        print(f"{game_name}: {per_game_game_accuracy[game_name]:.4f}")

if __name__ == '__main__':
    main()
