"""Targeted collection: extra images of rare classes, added to dataset_v2."""
import os, math, random
from PIL import Image
from ai2thor.controller import Controller

CLASSES = [
    "Apple", "Mug", "Bowl", "Bread", "Tomato", "Knife", "Plate", "Cup", "Pot",
    "RemoteControl", "Book", "Laptop", "Pillow", "Vase", "Box", "CellPhone",
    "Sofa", "ArmChair", "CoffeeTable", "Television",
]
# how many extra images to collect for each rare class
NEED = {"CellPhone": 90, "Book": 60, "RemoteControl": 50, "Knife": 30}
SCENES = [f"FloorPlan{i}" for i in range(1, 31)] + [f"FloorPlan{i}" for i in range(201, 231)]
OUT = "dataset_v2"
W, H = 640, 480
MIN_BOX = 10
VIEWS_PER_OBJECT = 3

random.seed(1)
counts = {c: 0 for c in CLASSES}
saved = 0
c = Controller(scene=SCENES[0], width=W, height=H, renderInstanceSegmentation=True)

for scene in SCENES:
    if all(n <= 0 for n in NEED.values()):
        break
    c.reset(scene=scene)
    c.step(action="InitialRandomSpawn", randomSeed=random.randint(0, 1000),
           forceVisible=True, numPlacementAttempts=5)
    positions = c.step(action="GetReachablePositions").metadata["actionReturn"]
    targets = [o for o in c.last_event.metadata["objects"] if NEED.get(o["objectType"], 0) > 0]

    for obj in targets:
        t = obj["objectType"]
        ox, oz = obj["position"]["x"], obj["position"]["z"]
        near = [p for p in positions if 0.5 <= math.dist((p["x"], p["z"]), (ox, oz)) <= 1.5]
        for _ in range(VIEWS_PER_OBJECT):
            if not near or NEED[t] <= 0:
                break
            p = random.choice(near)
            yaw = math.degrees(math.atan2(ox - p["x"], oz - p["z"])) + random.uniform(-15, 15)
            e = c.step(action="TeleportFull", position=p, rotation=dict(x=0, y=yaw, z=0),
                       horizon=random.choice([15, 30, 45]), standing=True)
            if not e.metadata["lastActionSuccess"]:
                continue
            lines, types = [], set()
            for oid, (x1, y1, x2, y2) in (e.instance_detections2D or {}).items():
                ot = oid.split("|")[0]
                if ot not in CLASSES or (x2 - x1) < MIN_BOX or (y2 - y1) < MIN_BOX:
                    continue
                lines.append(f"{CLASSES.index(ot)} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                             f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
                types.add(ot)
                counts[ot] += 1
            if t not in types:  # the target isn't clearly visible: skip this view
                continue
            name = f"boost_{scene}_{saved:04d}"
            Image.fromarray(e.frame).save(f"{OUT}/images/{name}.jpg")
            with open(f"{OUT}/labels/{name}.txt", "w") as f:
                f.write("\n".join(lines))
            saved += 1
            NEED[t] -= 1
    print(scene, "-> total extra images:", saved)

c.stop()
print(f"\nExtra images: {saved}")
print("Still missing:", {k: v for k, v in NEED.items() if v > 0} or "none")
print("Extra boxes for rare classes:")
for name in ["CellPhone", "Book", "RemoteControl", "Knife"]:
    print(f"  {name:14s} +{counts[name]}")
