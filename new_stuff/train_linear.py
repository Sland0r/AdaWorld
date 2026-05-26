import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import glob
import random
from collections import defaultdict
from pathlib import Path
import os
from tqdm import tqdm


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


def _resolve_dump_base_dir(dump_dir_idx):
    base_name = 'latent_actions_dump' if dump_dir_idx == 1 else 'latent_actions_dump_2'
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


def load_data(test_ratio=0.1, seed=42, dataset='both', dump_dir_idx=1):
    base_name, base_dir = _resolve_dump_base_dir(dump_dir_idx)
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

    # Global split: take approximately `test_ratio` fraction of all samples as test.
    all_samples = []
    for game_name in unique_games:
        all_samples.extend(samples_by_game[game_name])

    total_samples = len(all_samples)
    if total_samples < 2:
        raise RuntimeError('Need at least two total samples to create a train/test split.')

    rng = random.Random(seed)
    rng.shuffle(all_samples)
    test_count = max(1, int(round(total_samples * test_ratio)))
    test_count = min(test_count, total_samples - 1)

    test_samples = all_samples[:test_count]
    train_samples = all_samples[test_count:]

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


def evaluate(model, loader, action_mode, unique_games, device, action_names=None):
    model.eval()
    total_correct = 0.0
    total_count = 0
    per_game_correct = {game_name: 0.0 for game_name in unique_games}
    per_game_count = {game_name: 0 for game_name in unique_games}
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_z, batch_actions, batch_games in loader:
            batch_z = batch_z.to(device)
            batch_actions_device = batch_actions.to(device)
            logits = model(batch_z)
            batch_correct = _accuracy_from_logits(logits, batch_actions_device, action_mode).cpu()
            batch_actions_cpu = batch_actions.cpu()
            total_correct += batch_correct.sum().item()
            total_count += batch_correct.numel()

            if action_mode == 'multiclass':
                y_true.extend(batch_actions_cpu.view(-1).tolist())
                y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            else:
                y_true.append(batch_actions_cpu)
                y_pred.append((torch.sigmoid(logits.cpu()) >= 0.5).to(batch_actions_cpu.dtype))

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

    from sklearn.metrics import precision_recall_fscore_support

    if action_mode == 'multiclass':
        labels = sorted(set(y_true) | set(y_pred))
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', labels=labels, zero_division=0
        )
        p_class, r_class, f1_class, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=labels, zero_division=0
        )
        per_action_metrics = []
        for idx, action_idx in enumerate(labels):
            action_name = action_names.get(action_idx, f"A{action_idx}") if action_names else f"A{action_idx}"
            per_action_metrics.append({
                'action_idx': action_idx,
                'action_name': action_name,
                'precision': float(p_class[idx]),
                'recall': float(r_class[idx]),
                'f1': float(f1_class[idx]),
                'support': int(support[idx]),
            })
    else:
        y_true = torch.cat(y_true, dim=0).numpy() if y_true else []
        y_pred = torch.cat(y_pred, dim=0).numpy() if y_pred else []
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        p_class, r_class, f1_class, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        per_action_metrics = []
        for idx in range(len(p_class)):
            action_name = action_names.get(idx, f"A{idx}") if action_names else f"A{idx}"
            per_action_metrics.append({
                'action_idx': idx,
                'action_name': action_name,
                'precision': float(p_class[idx]),
                'recall': float(r_class[idx]),
                'f1': float(f1_class[idx]),
                'support': int(support[idx]),
            })

    summary = {
        'accuracy': total_accuracy,
        'precision_micro': float(p_micro),
        'recall_micro': float(r_micro),
        'f1_micro': float(f1_micro),
        'precision_macro': float(p_macro),
        'recall_macro': float(r_macro),
        'f1_macro': float(f1_macro),
    }
    return summary, per_game_accuracy, per_action_metrics


def train_multiclass_model(model, loader, criterion, optimizer, epochs, device, target_index=1, mask=None):
    for epoch in tqdm(range(epochs), desc="Training", file=sys.stdout):
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
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}", flush=True)


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

def load_data_per_game(test_ratio=0.1, seed=42, dataset='both', dump_dir_idx=1):
    """Load data grouped by game. Returns dict: game_name -> (train_dataset, test_dataset, num_actions, action_mode)."""
    base_name, base_dir = _resolve_dump_base_dir(dump_dir_idx)
    if dataset == 'both':
        files = glob.glob(os.path.join(base_dir, "**", "latent_actions.pt"), recursive=True)
    else:
        files = glob.glob(os.path.join(base_dir, dataset, "**", "latent_actions.pt"), recursive=True)
    files = sorted(files)
    if not files:
        raise RuntimeError(f'No latent_actions.pt files found under {base_name}/ ({base_dir}).')

    samples_by_game = defaultdict(list)

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
            actions = _extract_actions(data, z.shape[0], f)
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

    return game_datasets


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

        for epoch in tqdm(range(args.epochs), desc=f"  Training {game_name}", file=sys.stdout):
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
                tqdm.write(f"  Epoch {epoch+1}/{args.epochs}, Loss: {total_loss / len(train_loader):.4f}")

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
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Fraction of samples to use for test (per-game). Default 0.1')
    parser.add_argument('--dump-dir', type=int, choices=[1,2], default=1, dest='dump_dir',
                        help='Which dump dir to use: 1 -> latent_actions_dump, 2 -> latent_actions_dump_2')
    parser.add_argument('--per_game', action='store_true',
                        help='Train a separate model for each game instead of a single shared model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    if args.per_game:
        print("Loading per-game data...")
        game_datasets = load_data_per_game(test_ratio=args.test_ratio, dataset=args.dataset, dump_dir_idx=args.dump_dir)
        print(f"Games: {list(game_datasets.keys())}")
        per_game_results = train_per_game(game_datasets, args, device)
        print(f"\n{'='*60}")
        print("Per-game accuracy summary:")
        for game_name, acc in per_game_results.items():
            print(f"  {game_name}: {acc:.4f}")
        mean_acc = sum(per_game_results.values()) / len(per_game_results) if per_game_results else 0.0
        print(f"  Mean: {mean_acc:.4f}")
        return

    print("Loading data...", flush=True)
    train_dataset, test_dataset, num_actions, unique_games, action_mode, desc_map = load_data(test_ratio=args.test_ratio, dataset=args.dataset, dump_dir_idx=args.dump_dir)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}", flush=True)
    print(f"Hidden layers (actions): {args.action_hidden_layers}, Hidden layers (game): {args.game_hidden_layers}", flush=True)
    print(f"Games: {unique_games}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print('Dataloader sizes:', len(train_loader), len(test_loader), flush=True)
    print('Masking dimensions:', args.mask, flush=True)

    in_dim = train_dataset.tensors[0].shape[1]
    epochs = args.epochs

    model = build_mlp(in_dim, num_actions, args.action_hidden_layers).to(device)
    criterion = nn.CrossEntropyLoss() if action_mode == 'multiclass' else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Training started...", flush=True)
    for epoch in tqdm(range(epochs), desc="Training", file=sys.stdout):
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
            tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    print("Testing started...")
    metrics, per_game_accuracy, per_action_metrics = evaluate(
        model,
        test_loader,
        action_mode,
        unique_games,
        device,
        action_names=desc_map,
    )
    print(f"Total accuracy: {metrics['accuracy']:.4f}")
    print(f"Micro P/R/F1: {metrics['precision_micro']:.4f} / {metrics['recall_micro']:.4f} / {metrics['f1_micro']:.4f}")
    print(f"Macro P/R/F1: {metrics['precision_macro']:.4f} / {metrics['recall_macro']:.4f} / {metrics['f1_macro']:.4f}")
    print("\nPer-action metrics:")
    for action_metrics in per_action_metrics:
        print(
            f"  {action_metrics['action_name']} (A{action_metrics['action_idx']}): "
            f"P/R/F1 {action_metrics['precision']:.4f} / {action_metrics['recall']:.4f} / {action_metrics['f1']:.4f} "
            f"(support={action_metrics['support']})"
        )
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
