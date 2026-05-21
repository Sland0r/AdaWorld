import json
import os
from collections import defaultdict

data_dir = "/scratch-shared/scur0531/skipped_frames_v0.0.0"
action_game_count = defaultdict(set)
games = os.listdir(data_dir)

all_ok = True

for game in games:
    game_path = os.path.join(data_dir, game)
    if not os.path.isdir(game_path):
        continue
    
    # Try to find an actions.json
    actions_file = None
    for root, dirs, files in os.walk(game_path):
        if "actions.json" in files:
            actions_file = os.path.join(root, "actions.json")
            break
            
    if not actions_file:
        continue
        
    try:
        with open(actions_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {actions_file}: {e}")
        continue
        
    action_vocab = data.get("action_vocab", [])
    # Flatten action_vocab for display names:
    vocab_names = []
    # Index 0 is typically NOOP
    for v in action_vocab:
        vocab_names.append(v[0])
        
    actions = data.get("actions", [])
    
    # verify 16 + 4 pattern
    # 16 NOOPs
    noop = [0] * len(actions[0]['action']) if len(actions) > 0 else []
    
    is_pattern_valid = True
    current_idx = 0
    unique_actions_in_game = set()
    
    while current_idx < len(actions):
        # check 16 NOOPs (if enough frames left, or it might cut off)
        # actually, let's just observe the blocks of length 20
        block = actions[current_idx:current_idx+20]
        if len(block) == 20:
            for i in range(16):
                if block[i]['action'] != noop:
                    is_pattern_valid = False
            
            action_val = block[16]['action']
            for i in range(16, 20):
                if block[i]['action'] != action_val:
                    is_pattern_valid = False
            
            if is_pattern_valid and action_val != noop:
                # Find the index of the 1 to get the name
                try:
                    active_idx = action_val.index(1)
                    # The name is active_idx + 1 in vocab_names since NOOP is at 0
                    if active_idx + 1 < len(vocab_names):
                        action_name = vocab_names[active_idx + 1]
                        unique_actions_in_game.add(action_name)
                except ValueError:
                    pass
                    
        current_idx += 20
        
    if not is_pattern_valid:
        print(f"Pattern invalid in game: {game}")
        all_ok = False
        
    # Also add them to the global count mapping (action -> set of games)
    # Actually wait - does the user want the generic vocab name like "RIGHT", "ACTION_PRIMARY"
    # or the specific description like "sword"? The vocab name is standard across games.
    for act in unique_actions_in_game:
        action_game_count[act].add(game)

print(f"Pattern (16 NOOP + 4 Action) holds for all games? {all_ok}")
print("Action counts (number of games each action appears in):")
for act, games_set in sorted(action_game_count.items(), key=lambda x: -len(x[1])):
    print(f"{act}: {len(games_set)}")

