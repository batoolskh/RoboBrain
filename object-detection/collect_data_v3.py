"""Collect YOLO v3 data: 30 classes (v2 + kitchen appliances and surfaces)."""
import os, random
from collections import Counter
from PIL import Image
from ai2thor.controller import Controller

CLASSES = [
    # v2 kitchen objects
    "Apple", "Mug", "Bowl", "Bread", "Tomato", "Knife", "Plate", "Cup", "Pot",
    # v2 living room objects + furniture
    "RemoteControl", "Book", "Laptop", "Pillow", "Vase", "Box", "CellPhone",
    "Sofa", "ArmChair", "CoffeeTable", "Television",
    # v3 new: kitchen appliances and surfaces
    "SinkBasin", "Faucet", "Fridge", "CounterTop", "StoveBurner",
    "Microwave", "Toaster", "CoffeeMachine", "GarbageCan", "DiningTable",
]
SCENES = [(f"FloorPlan{i}", 30) for i in range(1, 31)] + \
         [(f"FloorPlan{i}", 15) for i in range(201, 231)]   # (scene, images per scene)
OUT = "dataset_v3"
SIZE = 600
MIN_PX = 10   # ignore boxes smaller than this (width or height)

random.seed(0)
os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(f"{OUT}/labels", exist_ok=True)
counts, saved = Counter(), 0

c = Controller(scene="FloorPlan1", width=SIZE, height=SIZE, fieldOfView=90,
               renderInstanceSegmentation=True, gridSize=0.25)

for scene, n in SCENES:
    c.reset(scene=scene)
    c.step(action="InitialRandomSpawn", randomSeed=random.randint(0, 9999), forceVisible=True)
    positions = c.step(action="GetReachablePositions").metadata["actionReturn"]
    got = 0
    for _ in range(n * 4):          # extra tries: some views have no objects
        if got == n:
            break
        e = c.step(action="TeleportFull", position=random.choice(positions),
                   rotation=dict(x=0, y=random.choice(range(0, 360, 30)), z=0),
                   horizon=random.choice([0, 15, 30, 45]), standing=True)
        if not e.metadata["lastActionSuccess"]:
            continue
        types = {o["objectId"]: o["objectType"] for o in e.metadata["objects"]}
        lines = []
        for oid, (x1, y1, x2, y2) in e.instance_detections2D.items():
            t = types.get(oid)
            if t not in CLASSES or x2 - x1 < MIN_PX or y2 - y1 < MIN_PX:
                continue
            xc, yc = (x1 + x2) / 2 / SIZE, (y1 + y2) / 2 / SIZE
            w, h = (x2 - x1) / SIZE, (y2 - y1) / SIZE
            lines.append(f"{CLASSES.index(t)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            counts[t] += 1
        if not lines:
            continue
        name = f"{scene}_{got:03d}"
        Image.fromarray(e.frame).save(f"{OUT}/images/{name}.jpg")
        with open(f"{OUT}/labels/{name}.txt", "w") as f:
            f.write("\n".join(lines))
        got += 1
        saved += 1
    print(f"{scene}: {got} images")

with open(f"{OUT}/data.yaml", "w") as f:
    f.write(f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")

print(f"\nSaved {saved} images. Boxes per class:")
for cls in CLASSES:
    print(f"  {cls:14s} {counts[cls]}")
c.stop()
