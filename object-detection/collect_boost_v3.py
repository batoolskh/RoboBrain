"""Boost rare classes for v3: close-up views from 0.5-1.5 m, incl. steep downward angles."""
import os, math, random
from collections import Counter
from PIL import Image
from ai2thor.controller import Controller

CLASSES = [
    "Apple", "Mug", "Bowl", "Bread", "Tomato", "Knife", "Plate", "Cup", "Pot",
    "RemoteControl", "Book", "Laptop", "Pillow", "Vase", "Box", "CellPhone",
    "Sofa", "ArmChair", "CoffeeTable", "Television",
    "SinkBasin", "Faucet", "Fridge", "CounterTop", "StoveBurner",
    "Microwave", "Toaster", "CoffeeMachine", "GarbageCan", "DiningTable",
]
RARE = ["CellPhone", "Book", "RemoteControl", "Knife", "Cup"]
SCENES = [f"FloorPlan{i}" for i in [*range(1, 31), *range(201, 231), *range(301, 331)]]
PER_OBJECT = 3      # close-up images per rare object
OUT, SIZE, MIN_PX = "dataset_v3", 600, 10

random.seed(1)
counts, saved = Counter(), 0
c = Controller(scene="FloorPlan1", width=SIZE, height=SIZE, fieldOfView=90,
               renderInstanceSegmentation=True, gridSize=0.25)


def save(e, name, must_see):
    """Write image + labels if the target object is visible. Returns True if saved."""
    types = {o["objectId"]: o["objectType"] for o in e.metadata["objects"]}
    det = e.instance_detections2D
    if must_see not in det:
        return False
    lines = []
    for oid, (x1, y1, x2, y2) in det.items():
        t = types.get(oid)
        if t not in CLASSES or x2 - x1 < MIN_PX or y2 - y1 < MIN_PX:
            continue
        lines.append(f"{CLASSES.index(t)} {(x1+x2)/2/SIZE:.6f} {(y1+y2)/2/SIZE:.6f} "
                     f"{(x2-x1)/SIZE:.6f} {(y2-y1)/SIZE:.6f}")
        counts[t] += 1
    Image.fromarray(e.frame).save(f"{OUT}/images/{name}.jpg")
    with open(f"{OUT}/labels/{name}.txt", "w") as f:
        f.write("\n".join(lines))
    return True


for scene in SCENES:
    c.reset(scene=scene)
    e = c.step(action="InitialRandomSpawn", randomSeed=random.randint(0, 9999), forceVisible=True)
    positions = c.step(action="GetReachablePositions").metadata["actionReturn"]
    targets = [o for o in c.last_event.metadata["objects"] if o["objectType"] in RARE]
    n_scene = 0
    for o in targets:
        p = o["position"]
        near = [q for q in positions if 0.5 < math.dist((q["x"], q["z"]), (p["x"], p["z"])) < 1.5]
        random.shuffle(near)
        got = 0
        for q in near:
            if got == PER_OBJECT:
                break
            yaw = math.degrees(math.atan2(p["x"] - q["x"], p["z"] - q["z"])) % 360
            e = c.step(action="TeleportFull", position=q, rotation=dict(x=0, y=yaw, z=0),
                       horizon=0, standing=True)
            if not e.metadata["lastActionSuccess"]:
                continue
            cam = e.metadata["cameraPosition"]
            pitch = math.degrees(math.atan2(cam["y"] - p["y"], math.dist((q["x"], q["z"]), (p["x"], p["z"]))))
            pitch = max(-30, min(60, pitch + random.uniform(-10, 10)))
            yaw += random.uniform(-20, 20)   # don't always center the object
            e = c.step(action="TeleportFull", position=q, rotation=dict(x=0, y=yaw, z=0),
                       horizon=pitch, standing=True)
            if save(e, f"boost_{scene}_{saved:04d}", o["objectId"]):
                got += 1
                saved += 1
                n_scene += 1
    print(f"{scene}: {n_scene} images")

print(f"\nSaved {saved} boost images. Boxes per class:")
for cls in CLASSES:
    if counts[cls]:
        print(f"  {cls:14s} {counts[cls]}")
c.stop()
