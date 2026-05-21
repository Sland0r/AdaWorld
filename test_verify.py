import json

data=json.load(open('/scratch-shared/scur0531/skipped_frames_v0.0.0/retro_8eyes-nes_v0.0.0/000000/000000/actions.json'))
actions = data['actions']

for i in range(40):
    print(actions[i]['action'])
