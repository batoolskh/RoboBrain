import os, random
from ai2thor.controller import Controller
from ultralytics import YOLO

model = YOLO("best.pt")
os.makedirs("yolo_test", exist_ok=True)

c = Controller(scene="FloorPlan5", width=640, height=480)
c.step(action="InitialRandomSpawn", randomSeed=123, forceVisible=True)
positions = c.step(action="GetReachablePositions").metadata["actionReturn"]

for i in range(10):
    e = c.step(action="TeleportFull", position=random.choice(positions),
               rotation=dict(x=0, y=random.choice(range(0, 360, 30)), z=0),
               horizon=30, standing=True)
    r = model(e.frame[:, :, ::-1].copy(), verbose=False)
    r[0].save(filename=f"yolo_test/test_{i}.jpg")
    print(i, [model.names[int(b.cls)] for b in r[0].boxes])

c.stop()
print("Done!")
