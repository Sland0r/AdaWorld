import json
import os
from collections import defaultdict

data_dir = "/scratch-shared/scur0531/skipped_frames_v0.0.0"
action_desc_count = defaultdict(set)
generic_action_count = defaultdict(set)
games = os.listdir(data_dir)

for game in games:
    game_path = os.path.join(data_dir, game)
    if not os.path.isdir(game_path):
        continue
    
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
    except Exception:
        continue
        
    action_vocab = [v[0] for v in data.get("action_vocab", [])]
    # some vocab entries might be a list of strings, we took the first
    action_descriptions = data.get("action_descriptions", [])
    
    actions = data.get("actions", [])
    
    noop = [0] * len(actions[0]['action']) if len(actions) > 0 else []
    
    current_idx = 0
    while current_idx < len(actions):
        block = actions[current_idx:current_idx+20]
        if len(block) == 20:
            action_val = block[16]['action']
            if action_val != noop:
                try:
                    active_idx = action_val.index(1)
                    
                    if active_idx + 1 < len(action_vocab):
                        gen_name = action_vocab[active_idx + 1]
                        generic_action_count[gen_name].add(game)
                        
                    if active_idx + 1 < len(action_descriptions):
                        desc_name = action_descriptions[active_idx + 1]
                        if desc_name != 'none':
                            action_desc_count[desc_name].add(game)
                except ValueError:
                    pass
        current_idx += 20

print("Specific action descriptions counts:")
for act, games_set in sorted(action_desc_count.items(), key=lambda x: -len(x[1])):
    print(f"{act}: {len(games_set)}")

print("\nGeneric vocab counts:")
for act, games_set in sorted(generic_action_count.items(), key=lambda x: -len(x[1])):
    print(f"{act}: {len(games_set)}")

