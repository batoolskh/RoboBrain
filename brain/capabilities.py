"""Ask AI2-THOR what the robot can do with each object class the YOLO model knows."""
import json
from ai2thor.controller import Controller
from ultralytics import YOLO

MODEL = "../object-detection/models/best_v3.pt"
SCENES = ([f"FloorPlan{i}" for i in range(1, 31)] +      # kitchens
          [f"FloorPlan{i}" for i in range(201, 231)])     # living rooms

classes = list(YOLO(MODEL).names.values())
caps = {c: {"pickup": False, "toggle": False, "open": False, "rooms": set()} for c in classes}

c = Controller(scene=SCENES[0], width=300, height=300)
for scene in SCENES:
    room = "kitchen" if int(scene[9:]) < 100 else "living_room"
    for o in c.reset(scene=scene).metadata["objects"]:
        t = o["objectType"]
        if t in caps:
            caps[t]["pickup"] |= o["pickupable"]
            caps[t]["toggle"] |= o["toggleable"]
            caps[t]["open"] |= o["openable"]
            caps[t]["rooms"].add(room)
    print("scanned", scene)
c.stop()

for v in caps.values():
    v["rooms"] = sorted(v["rooms"])
with open("object_capabilities.json", "w") as f:
    json.dump(caps, f, indent=2)

print(f"\n{'Object':15s} pickup toggle open  rooms")
for t, v in sorted(caps.items()):
    print(f"{t:15s} {'✓' if v['pickup'] else '-':^6} {'✓' if v['toggle'] else '-':^6} "
          f"{'✓' if v['open'] else '-':^4}  {', '.join(v['rooms'])}")
