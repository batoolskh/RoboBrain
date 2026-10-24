import os, random, json, math
import numpy as np
from ai2thor.controller import Controller
from ultralytics import YOLO

MODEL = "../object-detection/models/best.pt"
W = H = 600          # صورة مربعة لتبسيط حسابات الكاميرا
FOV = 90             # زاوية رؤية الكاميرا
F = (W / 2) / math.tan(math.radians(FOV / 2))   # البعد البؤري بالـ pixels

def pixel_to_world(u, v, d, meta):
    """يحوّل نقطة بالصورة (u, v) مع عمقها d لإحداثيات بالغرفة"""
    cam = meta["cameraPosition"]
    pitch = math.radians(meta["agent"]["cameraHorizon"])   # ميلان الكاميرا لتحت
    yaw = math.radians(meta["agent"]["rotation"]["y"])     # اتجاه الروبوت
    # 1) إحداثيات بالنسبة للكاميرا
    xc = (u - W / 2) * d / F
    yc = -(v - H / 2) * d / F
    zc = d
    # 2) نصلّح ميلان الكاميرا
    ya = yc * math.cos(pitch) - zc * math.sin(pitch)
    za = yc * math.sin(pitch) + zc * math.cos(pitch)
    # 3) نصلّح اتجاه الروبوت ونضيف مكان الكاميرا
    xw = xc * math.cos(yaw) + za * math.sin(yaw)
    zw = -xc * math.sin(yaw) + za * math.cos(yaw)
    return cam["x"] + xw, cam["y"] + ya, cam["z"] + zw

model = YOLO(MODEL)
c = Controller(scene="FloorPlan5", width=W, height=H, fieldOfView=FOV,
               renderDepthImage=True)
c.step(action="InitialRandomSpawn", randomSeed=123, forceVisible=True)
positions = c.step(action="GetReachablePositions").metadata["actionReturn"]
os.makedirs("loc_test", exist_ok=True)
results = []

for i in range(10):
    e = c.step(action="TeleportFull", position=random.choice(positions),
               rotation=dict(x=0, y=random.choice(range(0, 360, 30)), z=0),
               horizon=30, standing=True)
    if not e.metadata["lastActionSuccess"]:
        continue
    r = model(e.frame[:, :, ::-1].copy(), verbose=False)[0]
    r.save(filename=f"loc_test/frame_{i}.jpg")

    for b in r.boxes:
        cls = model.names[int(b.cls)]
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        u, v = (x1 + x2) / 2, (y1 + y2) / 2
        # العمق: وسيط منطقة صغيرة بنص المربع (أدق من pixel وحدة)
        a1 = max(0, int(x1 + (x2 - x1) * 0.4)); a2 = min(W, int(x1 + (x2 - x1) * 0.6) + 1)
        b1 = max(0, int(y1 + (y2 - y1) * 0.4)); b2 = min(H, int(y1 + (y2 - y1) * 0.6) + 1)
        d = float(np.median(e.depth_frame[b1:b2, a1:a2]))
        xw, yw, zw = pixel_to_world(u, v, d, e.metadata)

        # المكان الحقيقي: أقرب غرض من نفس النوع
        cands = [o for o in e.metadata["objects"] if o["objectType"] == cls]
        if not cands:
            continue
        gt = min(cands, key=lambda o: (o["axisAlignedBoundingBox"]["center"]["x"] - xw) ** 2
                                     + (o["axisAlignedBoundingBox"]["center"]["z"] - zw) ** 2)
        g = gt["axisAlignedBoundingBox"]["center"]
        err = math.dist((xw, yw, zw), (g["x"], g["y"], g["z"]))
        results.append(dict(frame=i, cls=cls, est=[xw, yw, zw],
                            gt=[g["x"], g["y"], g["z"]], error_m=err))
        print(f"{i} {cls:7s} est=({xw:.2f}, {zw:.2f})  real=({g['x']:.2f}, {g['z']:.2f})  error={err*100:.0f} cm")
        ctr = lambda o: (o["axisAlignedBoundingBox"]["center"]["x"], o["axisAlignedBoundingBox"]["center"]["y"], o["axisAlignedBoundingBox"]["center"]["z"])
        near = min(e.metadata["objects"], key=lambda o: math.dist((xw, yw, zw), ctr(o)))
        print(f"      -> closest real object: {near['objectType']} ({math.dist((xw, yw, zw), ctr(near))*100:.0f} cm)")

c.stop()
with open("loc_results.json", "w") as f:
    json.dump(results, f, indent=2)
if results:
    print(f"\nAverage error: {np.mean([x['error_m'] for x in results]) * 100:.1f} cm over {len(results)} objects")



