import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import glob
import random
from collections import defaultdict
from pathlib import Path
import os


def build_mlp(in_dim, out_dim, n_hidden, hidden_dim=256):
    if n_hidden == 0:
        return nn.Linear(in_dim, out_dim)
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)

def _get_game_name(data, path):
    if data.get('game_name'):
        return data['game_name']
    parts = Path(path).parts
    # Fallback to path heuristics
    path_str = str(path)
    if 'latent_actions_dump_2' in path_str:
        return parts[-3] if len(parts) >= 3 else 'unknown'
    if 'latent_actions_dump' in path_str:
        return parts[-4] if len(parts) >= 4 else 'unknown'
    return parts[-2] if len(parts) >= 2 else 'unknown'
    


def _format_actions(actions, num_samples, file_path):
    desc_map = {}
    if isinstance(actions, list) and len(actions) > 0 and isinstance(actions[0], dict):
        if 'action' in actions[0]:
            for a in actions:
                if 'desc' in a:
                    try:
                        idx = a['action'].index(1)
                        desc_map[idx] = a['desc']
                    except ValueError:
                        pass
            actions = [a['action'] for a in actions]
    actions = torch.as_tensor(actions)
    if actions.ndim == 0:
        actions = actions.unsqueeze(0)
    if actions.shape[0] != num_samples:
        raise ValueError(
            f"Action count mismatch in {file_path}: z_mu has {num_samples} samples but actions has shape {tuple(actions.shape)}"
        )
    return actions, desc_map


def _resolve_dump_base_dir():
    base_name = 'latent_actions_skipped'
    repo_root = Path(__file__).resolve().parents[1]
    return base_name, str(repo_root / base_name)


def _extract_actions(data, num_samples, file_path):
    actions_raw = data.get('actions')
    has_dense_actions = False
    if actions_raw is not None:
        try:
            has_dense_actions = len(actions_raw) > 0
        except TypeError:
            has_dense_actions = True

    if has_dense_actions:
        actions, desc_map = _format_actions(actions_raw, num_samples, file_path)
        if actions.ndim == 2 and actions.shape[1] == 1 and torch.all(actions == actions.long().to(actions.dtype)):
            actions = actions.squeeze(1)
        return actions, desc_map

    keyboard_labels = data.get('keyboard_labels')
    if keyboard_labels is None:
        return None, {}

    keyboard_labels = torch.as_tensor(keyboard_labels)
    if keyboard_labels.ndim == 1:
        keyboard_labels = keyboard_labels.unsqueeze(0)
    if keyboard_labels.shape[0] != num_samples:
        raise ValueError(
            f"Action count mismatch in {file_path}: z_mu has {num_samples} samples "
            f"but keyboard_labels has shape {tuple(keyboard_labels.shape)}"
        )
    keyboard_labels = (keyboard_labels > 0).to(torch.float32)

    mouse_left_right = torch.zeros((num_samples, 2), dtype=torch.float32)
    mouse_buttons = data.get('mouse_buttons')
    if mouse_buttons is not None:
        mouse_buttons = torch.as_tensor(mouse_buttons)
        if mouse_buttons.ndim == 0:
            mouse_buttons = mouse_buttons.unsqueeze(0)
        if mouse_buttons.shape[0] != num_samples:
            raise ValueError(
                f"Action count mismatch in {file_path}: z_mu has {num_samples} samples "
                f"but mouse_buttons has shape {tuple(mouse_buttons.shape)}"
            )
        mouse_left_right[:, 0] = (mouse_buttons == 0).to(torch.float32)
        mouse_left_right[:, 1] = (mouse_buttons == 1).to(torch.float32)

    return torch.cat([keyboard_labels, mouse_left_right], dim=1), {}


def _build_dataset(samples):
    z = torch.stack([sample[0] for sample in samples], dim=0)
    actions = torch.stack([sample[1] for sample in samples], dim=0)
    games = [sample[2] for sample in samples]
    return z, actions, games


def _filter_and_remap_actions(samples, desc_map, exclude_actions):
    """Filter out samples whose action label is in exclude_actions and remap indices."""
    if not exclude_actions:
        return samples, desc_map

    # Build set of action indices to exclude based on description
    exclude_indices = set()
    for idx, desc in desc_map.items():
        if desc in exclude_actions:
            exclude_indices.add(idx)
    if not exclude_indices:
        print(f"Warning: none of {exclude_actions} matched any action descriptions in desc_map: {desc_map}")
        return samples, desc_map

    print(f"Excluding actions: {exclude_actions} (indices: {sorted(exclude_indices)})")

    # Filter samples: keep only those whose action index is not excluded
    filtered = []
    for sample in samples:
        action = sample[1]
        if action.ndim == 0 or (action.ndim == 1 and action.shape[0] == 1):
            action_idx = int(action.item())
        elif action.ndim == 1 and torch.all((action == 0) | (action == 1)) and action.sum() == 1:
            action_idx = int(action.argmax().item())
        else:
            action_idx = None
        if action_idx is not None and action_idx in exclude_indices:
            continue
        filtered.append(sample)

    print(f"Filtered {len(samples)} -> {len(filtered)} samples")

    # Build remapping: old index -> new contiguous index
    kept_indices = sorted(set(range(max(desc_map.keys()) + 1)) - exclude_indices) if desc_map else []
    remap = {old: new for new, old in enumerate(kept_indices)}
    new_desc_map = {remap[old]: desc for old, desc in desc_map.items() if old in remap}

    # Remap actions in filtered samples
    remapped = []
    for sample in filtered:
        action = sample[1]
        if action.ndim == 0 or (action.ndim == 1 and action.shape[0] == 1):
            old_idx = int(action.item())
            new_action = torch.tensor(remap[old_idx], dtype=action.dtype)
            remapped.append((sample[0], new_action) + sample[2:])
        elif action.ndim == 1 and torch.all((action == 0) | (action == 1)) and action.sum() == 1:
            old_idx = int(action.argmax().item())
            new_onehot = torch.zeros(len(kept_indices), dtype=action.dtype)
            new_onehot[remap[old_idx]] = 1
            remapped.append((sample[0], new_onehot) + sample[2:])
        else:
            remapped.append(sample)

    print(f"Kept actions: {new_desc_map}")
    return remapped, new_desc_map


def load_data(test_ratio=0.2, seed=42, dataset='both', exclude_actions=None):
    base_name, base_dir = _resolve_dump_base_dir()
    if dataset == 'both':
        files = glob.glob(os.path.join(base_dir, "**", "latent_actions.pt"), recursive=True)
    else:
        files = glob.glob(os.path.join(base_dir, dataset, "**", "latent_actions.pt"), recursive=True)
    files = sorted(files)
    if not files:
        raise RuntimeError(f'No latent_actions.pt files found under {base_name}/ ({base_dir}).')

    samples_by_game = defaultdict(list)
    unique_games = []
    global_desc_map = {}

    for f in files:
        try:
            data = torch.load(f, map_location='cpu')
        except Exception as e:
            print(f"Skipping {f}: failed to load ({e})")
            continue

        z = torch.as_tensor(data['z_mu'], dtype=torch.float32)
        if z.ndim == 1:
            z = z.unsqueeze(0)

        try:
            actions, desc_map = _extract_actions(data, z.shape[0], f)
            global_desc_map.update(desc_map)
        except (RuntimeError, TypeError, ValueError) as e:
            print(f"Skipping {f}: {e}")
            continue
        if actions is None:
            print(f"Skipping {f} because no actions were found.")
            continue

        game_name = _get_game_name(data, f)
        if game_name not in samples_by_game:
            unique_games.append(game_name)

        for sample_z, sample_action in zip(z, actions):
            samples_by_game[game_name].append((sample_z, sample_action, game_name))

    if not samples_by_game:
        raise RuntimeError(
            f"No usable samples found under {base_name}/ ({base_dir}). "
            "All files were missing actions or unreadable."
        )

    # Filter and remap actions if exclude_actions is specified
    if exclude_actions:
        original_desc_map = dict(global_desc_map)
        for game_name in unique_games:
            samples_by_game[game_name], new_desc_map = _filter_and_remap_actions(
                samples_by_game[game_name], original_desc_map, exclude_actions
            )
        global_desc_map = new_desc_map
        # Remove games that have no samples left after filtering
        unique_games = [g for g in unique_games if len(samples_by_game[g]) >= 2]
        if not unique_games:
            raise RuntimeError("No games with enough samples remaining after action filtering.")

    min_count = min(len(samples_by_game[g]) for g in unique_games)
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
    elif torch.all(train_actions.sum(dim=1) == 1) and torch.all((train_actions == 0) | (train_actions == 1)):
        # one-hot → class indices
        train_actions = train_actions.argmax(dim=1)
        test_actions = test_actions.argmax(dim=1)
        num_actions = int(max(train_actions.max().item(), test_actions.max().item())) + 1
        action_mode = 'multiclass'
    else:
        num_actions = train_actions.shape[1]
        action_mode = 'multilabel'

    game_to_idx = {game_name: idx for idx, game_name in enumerate(unique_games)}
    train_games = torch.tensor([game_to_idx[game_name] for game_name in train_games], dtype=torch.long)
    test_games = torch.tensor([game_to_idx[game_name] for game_name in test_games], dtype=torch.long)

    train_dataset = TensorDataset(train_z, train_actions, train_games)
    test_dataset = TensorDataset(test_z, test_actions, test_games)

    return train_dataset, test_dataset, num_actions, unique_games, action_mode, global_desc_map


def _accuracy_from_logits(logits, targets, action_mode):
    if action_mode == 'multiclass':
        predictions = logits.argmax(dim=1)
        return (predictions == targets.long()).float()

    predictions = (torch.sigmoid(logits) >= 0.5).to(targets.dtype)
    return (predictions == targets).view(targets.size(0), -1).float().mean(dim=1)


def evaluate(model, loader, action_mode, unique_games, device):
    model.eval()
    total_correct = 0.0
    total_count = 0
    per_game_correct = {game_name: 0.0 for game_name in unique_games}
    per_game_count = {game_name: 0 for game_name in unique_games}
    per_action_correct = {}
    per_action_count = {}

    with torch.no_grad():
        for batch_z, batch_actions, batch_games in loader:
            batch_z = batch_z.to(device)
            batch_actions = batch_actions.to(device)
            logits = model(batch_z)
            batch_correct = _accuracy_from_logits(logits, batch_actions, action_mode).cpu()
            batch_actions = batch_actions.cpu()
            total_correct += batch_correct.sum().item()
            total_count += batch_correct.numel()

            for game_idx in batch_games.unique(sorted=True):
                game_mask = batch_games == game_idx
                game_name = unique_games[game_idx.item()]
                per_game_correct[game_name] += batch_correct[game_mask].sum().item()
                per_game_count[game_name] += game_mask.sum().item()

            # Track per-action accuracy
            for action_idx in batch_actions.unique(sorted=True):
                action_mask = batch_actions == action_idx
                action_key = action_idx.item() if batch_actions.ndim == 1 else tuple(action_idx.cpu().numpy())
                if action_key not in per_action_correct:
                    per_action_correct[action_key] = 0.0
                    per_action_count[action_key] = 0
                per_action_correct[action_key] += batch_correct[action_mask].sum().item()
                per_action_count[action_key] += action_mask.sum().item()

    total_accuracy = total_correct / total_count if total_count else 0.0
    per_game_accuracy = {
        game_name: (per_game_correct[game_name] / per_game_count[game_name] if per_game_count[game_name] else 0.0)
        for game_name in unique_games
    }
    per_action_accuracy = {
        action_key: (per_action_correct[action_key] / per_action_count[action_key] if per_action_count[action_key] else 0.0)
        for action_key in per_action_correct.keys()
    }
    return total_accuracy, per_game_accuracy, per_action_accuracy


def train_multiclass_model(model, loader, criterion, optimizer, epochs, device, target_index=1, mask=None):
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_z, batch_targets, batch_games in loader:
            batch_z = batch_z.to(device)
            if mask:
                batch_z = batch_z.clone()
                batch_z[:, mask] = 0.0
            if target_index == 1:
                targets = batch_targets.to(device)
            elif target_index == 2:
                targets = batch_games.to(device)
            else:
                raise ValueError(f"Unsupported target_index: {target_index}")

            optimizer.zero_grad()
            logits = model(batch_z)
            loss = criterion(logits, targets.long().view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}")


def evaluate_multiclass_model(model, loader, unique_games, device, target_index=1):
    model.eval()
    total_correct = 0.0
    total_count = 0
    per_game_correct = {game_name: 0.0 for game_name in unique_games}
    per_game_count = {game_name: 0 for game_name in unique_games}

    with torch.no_grad():
        for batch_z, batch_targets, batch_games in loader:
            batch_z = batch_z.to(device)
            if target_index == 1:
                targets = batch_targets.to(device)
            elif target_index == 2:
                targets = batch_games.to(device)
            else:
                raise ValueError(f"Unsupported target_index: {target_index}")

            logits = model(batch_z)
            predictions = logits.argmax(dim=1)
            batch_correct = (predictions == targets.long().view(-1)).float().cpu()
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

def load_data_per_game(test_ratio=0.2, seed=42, dataset='both', exclude_actions=None):
    """Load data grouped by game. Returns dict: game_name -> (train_dataset, test_dataset, num_actions, action_mode)."""
    base_name, base_dir = _resolve_dump_base_dir()
    if dataset == 'both':
        files = glob.glob(os.path.join(base_dir, "**", "latent_actions.pt"), recursive=True)
    else:
        files = glob.glob(os.path.join(base_dir, dataset, "**", "latent_actions.pt"), recursive=True)
    files = sorted(files)
    if not files:
        raise RuntimeError(f'No latent_actions.pt files found under {base_name}/ ({base_dir}).')

    samples_by_game = defaultdict(list)
    global_desc_map = {}

    for f in files:
        try:
            data = torch.load(f, map_location='cpu')
        except Exception as e:
            print(f"Skipping {f}: failed to load ({e})")
            continue

        z = torch.as_tensor(data['z_mu'], dtype=torch.float32)
        if z.ndim == 1:
            z = z.unsqueeze(0)

        try:
            actions, desc_map = _extract_actions(data, z.shape[0], f)
            global_desc_map.update(desc_map)
        except (RuntimeError, TypeError, ValueError) as e:
            print(f"Skipping {f}: {e}")
            continue
        if actions is None:
            print(f"Skipping {f} because no actions were found.")
            continue

        game_name = _get_game_name(data, f)

        for sample_z, sample_action in zip(z, actions):
            samples_by_game[game_name].append((sample_z, sample_action))

    if not samples_by_game:
        raise RuntimeError(
            f"No usable samples found under {base_name}/ ({base_dir}). "
            "All files were missing actions or unreadable."
        )

    # Filter and remap actions if exclude_actions is specified
    if exclude_actions:
        original_desc_map = dict(global_desc_map)
        for game_name in list(samples_by_game.keys()):
            samples_by_game[game_name], new_desc_map = _filter_and_remap_actions(
                samples_by_game[game_name], original_desc_map, exclude_actions
            )
        global_desc_map = new_desc_map

    rng = random.Random(seed)
    game_datasets = {}

    for game_name, samples in samples_by_game.items():
        if len(samples) < 2:
            print(f"Skipping {game_name}: only {len(samples)} sample(s).")
            continue

        game_samples = list(samples)
        rng.shuffle(game_samples)

        test_count = max(1, int(round(len(game_samples) * test_ratio)))
        test_count = min(test_count, len(game_samples) - 1)

        test_s = game_samples[:test_count]
        train_s = game_samples[test_count:]

        train_z = torch.stack([s[0] for s in train_s])
        test_z = torch.stack([s[0] for s in test_s])
        train_actions = torch.stack([s[1] for s in train_s])
        test_actions = torch.stack([s[1] for s in test_s])

        if train_actions.ndim == 1:
            num_actions = int(torch.max(torch.cat([train_actions, test_actions])).item()) + 1
            action_mode = 'multiclass'
        elif torch.all(train_actions.sum(dim=1) == 1) and torch.all((train_actions == 0) | (train_actions == 1)):
            train_actions = train_actions.argmax(dim=1)
            test_actions = test_actions.argmax(dim=1)
            num_actions = int(max(train_actions.max().item(), test_actions.max().item())) + 1
            action_mode = 'multiclass'
        else:
            num_actions = train_actions.shape[1]
            action_mode = 'multilabel'

        game_datasets[game_name] = (
            TensorDataset(train_z, train_actions),
            TensorDataset(test_z, test_actions),
            num_actions,
            action_mode,
        )

    return game_datasets, global_desc_map


def train_per_game(game_datasets, args, device):
    """Train and evaluate a separate model for each game. Returns dict: game_name -> test_accuracy."""
    results = {}

    for game_name, (train_dataset, test_dataset, num_actions, action_mode) in game_datasets.items():
        print(f"\n{'='*60}")
        print(f"Game: {game_name}  |  train={len(train_dataset)}  test={len(test_dataset)}  actions={num_actions}  mode={action_mode}")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        in_dim = train_dataset.tensors[0].shape[1]
        model = build_mlp(in_dim, num_actions, args.action_hidden_layers).to(device)
        criterion = nn.CrossEntropyLoss() if action_mode == 'multiclass' else nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for batch_z, batch_actions in train_loader:
                batch_z = batch_z.to(device)
                if args.mask:
                    batch_z = batch_z.clone()
                    batch_z[:, args.mask] = 0.0
                batch_actions = batch_actions.to(device)
                optimizer.zero_grad()
                pred = model(batch_z)
                if action_mode == 'multiclass':
                    loss = criterion(pred, batch_actions.long().view(-1))
                else:
                    loss = criterion(pred, batch_actions.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}, Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        total_correct = 0.0
        total_count = 0
        with torch.no_grad():
            for batch_z, batch_actions in test_loader:
                batch_z = batch_z.to(device)
                batch_actions = batch_actions.to(device)
                logits = model(batch_z)
                if action_mode == 'multiclass':
                    batch_correct = (logits.argmax(dim=1) == batch_actions.long().view(-1)).float()
                else:
                    batch_correct = ((torch.sigmoid(logits) >= 0.5).to(batch_actions.dtype) == batch_actions) \
                        .view(batch_actions.size(0), -1).float().mean(dim=1)
                total_correct += batch_correct.sum().item()
                total_count += batch_correct.numel()

        accuracy = total_correct / total_count if total_count else 0.0
        results[game_name] = accuracy
        print(f"  Test accuracy: {accuracy:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--action_hidden_layers', type=int, default=1)
    parser.add_argument('--game_hidden_layers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--mask', type=int, nargs='*', default=[],
                        help='Dimensions to zero out during training (e.g. --mask 0 3 5)')
    parser.add_argument('--dataset', type=str, default='both', choices=['adaworld', 'olafworld', 'both'],
                        help='Which dataset to train on (default: both)')
    parser.add_argument('--exclude_actions', type=str, nargs='*', default=[],
                        help='Action descriptions to exclude (e.g. --exclude_actions up shoot)')
    parser.add_argument('--per_game', action='store_true',
                        help='Train a separate model for each game instead of a single shared model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.per_game:
        print("Loading per-game data...")
        game_datasets, desc_map = load_data_per_game(dataset=args.dataset, exclude_actions=args.exclude_actions or None)
        print(f"Games: {list(game_datasets.keys())}")
        per_game_results = train_per_game(game_datasets, args, device)
        print(f"\n{'='*60}")
        print("Per-game accuracy summary:")
        for game_name, acc in per_game_results.items():
            print(f"  {game_name}: {acc:.4f}")
        mean_acc = sum(per_game_results.values()) / len(per_game_results) if per_game_results else 0.0
        print(f"  Mean: {mean_acc:.4f}")
        return

    print("Loading data...")
    train_dataset, test_dataset, num_actions, unique_games, action_mode, desc_map = load_data(dataset=args.dataset, exclude_actions=args.exclude_actions or None)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Hidden layers (actions): {args.action_hidden_layers}, Hidden layers (game): {args.game_hidden_layers}")
    print(f"Games: {unique_games}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print('Dataloader sizes:', len(train_loader), len(test_loader))
    print('Masking dimensions:', args.mask)

    in_dim = train_dataset.tensors[0].shape[1]
    epochs = args.epochs

    model = build_mlp(in_dim, num_actions, args.action_hidden_layers).to(device)
    criterion = nn.CrossEntropyLoss() if action_mode == 'multiclass' else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Training started...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_z, batch_actions, _ in train_loader:
            batch_z = batch_z.to(device)
            if args.mask:
                batch_z = batch_z.clone()
                batch_z[:, args.mask] = 0.0
            batch_actions = batch_actions.to(device)
            optimizer.zero_grad()
            pred = model(batch_z)
            if action_mode == 'multiclass':
                loss = criterion(pred, batch_actions.long().view(-1))
            else:
                loss = criterion(pred, batch_actions.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    print("Testing started...")
    total_accuracy, per_game_accuracy, per_action_accuracy = evaluate(model, test_loader, action_mode, unique_games, device)
    print(f"Total accuracy: {total_accuracy:.4f}")
    print("\nPer-action accuracy:")
    for action_key in sorted(per_action_accuracy.keys()):
        label = desc_map.get(action_key, f"A{action_key}")
        print(f"  {label}: {per_action_accuracy[action_key]:.4f}")
    # for game_name in unique_games:
    #     print(f"{game_name}: {per_game_accuracy[game_name]:.4f}")

    print("Training game predictor...")
    game_model = build_mlp(in_dim, len(unique_games), args.game_hidden_layers).to(device)
    game_criterion = nn.CrossEntropyLoss()
    game_optimizer = optim.Adam(game_model.parameters(), lr=1e-3)
    train_multiclass_model(game_model, train_loader, game_criterion, game_optimizer, epochs, device, target_index=2, mask=args.mask)

    print("Testing game predictor...")
    game_accuracy, per_game_game_accuracy = evaluate_multiclass_model(game_model, test_loader, unique_games, device, target_index=2)
    print(f"Game accuracy: {game_accuracy:.4f}")
    # for game_name in unique_games:
    #     print(f"{game_name}: {per_game_game_accuracy[game_name]:.4f}")

if __name__ == '__main__':
    main()
