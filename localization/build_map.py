"""Explore a scene, build a fused object map, and evaluate it against ground truth."""
import os, json, math, random
import numpy as np
from ai2thor.controller import Controller
from ultralytics import YOLO
from localizer import ObjectLocalizer, ObjectMap

CONFIG = {
    "scene": "FloorPlan5",
    "model": "../object-detection/models/best.pt",
    "size": 600,           # square images keep the camera math simple
    "fov": 90,
    "n_views": 30,         # random viewpoints to explore
    "seed": 42,            # same seed = same results every run
    "conf": 0.5,           # ignore YOLO detections below this confidence
    "merge_dist": 0.3,     # meters
    "min_sightings": 2,
    "out_dir": "output",
}


def center(obj):
    c = obj["axisAlignedBoundingBox"]["center"]
    return (c["x"], c["y"], c["z"])


def main(cfg):
    random.seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)

    localizer = ObjectLocalizer(YOLO(cfg["model"]), cfg["size"], cfg["size"], cfg["fov"], cfg["conf"])
    obj_map = ObjectMap(cfg["merge_dist"], cfg["min_sightings"])

    c = Controller(scene=cfg["scene"], width=cfg["size"], height=cfg["size"],
                   fieldOfView=cfg["fov"], renderDepthImage=True)
    c.step(action="InitialRandomSpawn", randomSeed=cfg["seed"], forceVisible=True)
    positions = c.step(action="GetReachablePositions").metadata["actionReturn"]

    # 1) Explore: look around from random viewpoints and collect detections
    for _ in range(cfg["n_views"]):
        e = c.step(action="TeleportFull", position=random.choice(positions),
                   rotation=dict(x=0, y=random.choice(range(0, 360, 30)), z=0),
                   horizon=30, standing=True)
        if not e.metadata["lastActionSuccess"]:
            continue
        detections, _ = localizer.localize(e)
        for d in detections:
            obj_map.add(d)
    ground_truth = c.last_event.metadata["objects"]
    c.stop()

    # 2) Evaluate confirmed objects against the simulator's ground truth
    report = []
    print(f"\n{'Object':8s} {'Seen':>4s} {'Conf':>5s}   {'Position (x, z)':>16s}   {'Error':>6s}")
    for o in sorted(obj_map.confirmed(), key=lambda o: o["cls"]):
        cands = [g for g in ground_truth if g["objectType"] == o["cls"]]
        err = min(math.dist(o["position"], center(g)) for g in cands) if cands else None
        x, _, z = o["position"]
        err_txt = f"{err*100:.0f} cm" if err is not None else "n/a"
        print(f"{o['cls']:8s} {o['sightings']:4d} {o['conf']:5.2f}   ({x:6.2f}, {z:6.2f})   {err_txt:>6s}")
        report.append({**o, "error_m": err})

    errors = [r["error_m"] for r in report if r["error_m"] is not None]
    print(f"\nConfirmed objects: {len(report)}   "
          f"Rejected (seen < {cfg['min_sightings']}x): {len(obj_map.rejected())}")
    if errors:
        print(f"Mean error: {np.mean(errors)*100:.1f} cm   Median: {np.median(errors)*100:.1f} cm")

    with open(os.path.join(cfg["out_dir"], "object_map.json"), "w") as f:
        json.dump({"config": cfg, "objects": report}, f, indent=2)
    print(f"Saved map to {cfg['out_dir']}/object_map.json")


if __name__ == "__main__":
    main(CONFIG)
