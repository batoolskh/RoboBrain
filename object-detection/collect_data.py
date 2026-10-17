import os, random
from PIL import Image
from ai2thor.controller import Controller

CLASSES = ["Apple", "Mug", "Bowl", "Bread", "Tomato", "Knife"]
OUT = "dataset"
W, H = 640, 480
SCENES = [f"FloorPlan{i}" for i in range(1, 31)]
IMGS_PER_SCENE = 20

os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(f"{OUT}/labels", exist_ok=True)

c = Controller(scene=SCENES[0], width=W, height=H, renderInstanceSegmentation=True)

for scene in SCENES:
    c.reset(scene=scene)
    c.step(action="InitialRandomSpawn", randomSeed=random.randint(0, 1000),
           forceVisible=True, numPlacementAttempts=5)
    positions = c.step(action="GetReachablePositions").metadata["actionReturn"]
    saved, tries = 0, 0
    while saved < IMGS_PER_SCENE and tries < IMGS_PER_SCENE * 5:
        tries += 1
        e = c.step(action="TeleportFull", position=random.choice(positions),
                   rotation=dict(x=0, y=random.choice(range(0, 360, 30)), z=0),
                   horizon=random.choice([0, 15, 30, 45]), standing=True)
        if not e.metadata["lastActionSuccess"]:
            continue
        lines = []
        for oid, (x1, y1, x2, y2) in (e.instance_detections2D or {}).items():
            t = oid.split("|")[0]
            if t not in CLASSES or (x2 - x1) < 10 or (y2 - y1) < 10:
                continue
            lines.append(f"{CLASSES.index(t)} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                         f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
        if not lines:
            continue
        name = f"{scene}_{saved:03d}"
        Image.fromarray(e.frame).save(f"{OUT}/images/{name}.jpg")
        with open(f"{OUT}/labels/{name}.txt", "w") as f:
            f.write("\n".join(lines))
        saved += 1
    print(scene, "->", saved, "images")

c.stop()
with open(f"{OUT}/data.yaml", "w") as f:
    f.write(f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")
print("Done!")











































































































































































