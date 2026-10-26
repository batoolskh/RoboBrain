"""Fetch task: explore, find the target, plan with A*, drive there, and pick it up."""
import os, sys, math, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from ai2thor.controller import Controller
from ultralytics import YOLO

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "localization"))
from localizer import ObjectLocalizer, ObjectMap
from planner import GridMap, astar, path_to_actions, to_cell, to_xz

CONFIG = {
    "scene": "FloorPlan5",
    "target": "Apple",
    "model": "../object-detection/models/best.pt",
    "size": 600,
    "fov": 90,
    "n_views": 40,
    "seed": 42,
    "conf": 0.5,
    "verify_conf": 0.25,    # lower threshold when confirming a target we expect to see
    "reach": 1.0,           # max distance (m) from the robot to the object to grab it
    "min_start_dist": 2.0,  # start at least this far from the target (m)
    "max_attempts": 3,      # approach cells to try before giving up
    "out_dir": "output",
}


def explore(c, localizer, positions, n_views):
    """Look around from random viewpoints and build the object map."""
    obj_map = ObjectMap(merge_dist=0.3, min_sightings=2)
    for _ in range(n_views):
        e = c.step(action="TeleportFull", position=random.choice(positions),
                   rotation=dict(x=0, y=random.choice(range(0, 360, 30)), z=0),
                   horizon=30, standing=True)
        if e.metadata["lastActionSuccess"]:
            for d in localizer.localize(e)[0]:
                obj_map.add(d)
    return obj_map


def agent_cell(c):
    p = c.last_event.metadata["agent"]["position"]
    return to_cell(p["x"], p["z"])


def reset_view(c):
    """Snap the heading to the grid (0/90/180/270) and look straight ahead before driving."""
    m = c.last_event.metadata["agent"]
    heading = round(m["rotation"]["y"] / 90) * 90 % 360
    c.step(action="TeleportFull", position=m["position"],
           rotation=dict(x=0, y=heading, z=0), horizon=0, standing=True)
    return heading


def face(c, target_pos):
    """Turn and tilt the camera toward a 3D point."""
    m = c.last_event.metadata
    dx = target_pos[0] - m["agent"]["position"]["x"]
    dz = target_pos[2] - m["agent"]["position"]["z"]
    turn = (math.degrees(math.atan2(dx, dz)) - m["agent"]["rotation"]["y"] + 540) % 360 - 180
    c.step(action="RotateRight" if turn > 0 else "RotateLeft", degrees=abs(turn))
    m = c.last_event.metadata
    pitch = math.degrees(math.atan2(m["cameraPosition"]["y"] - target_pos[1], math.hypot(dx, dz)))
    pitch -= m["agent"]["cameraHorizon"]
    return c.step(action="LookDown" if pitch > 0 else "LookUp", degrees=abs(pitch))


def pick_up(c, localizer, target, est_pos):
    """Re-detect the target with YOLO, then grasp the matching object."""
    e = c.last_event
    r = localizer.model(e.frame[:, :, ::-1].copy(), conf=CONFIG["verify_conf"], verbose=False)[0]
    boxes = [b for b in r.boxes if localizer.model.names[int(b.cls)] == target]
    if boxes:  # refine the position with what we see right now
        b = max(boxes, key=lambda b: float(b.conf))
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        d = localizer.box_depth(e.depth_frame, x1, y1, x2, y2)
        if d:
            est_pos = localizer.pixel_to_world((x1 + x2) / 2, (y1 + y2) / 2, d, e.metadata)
    # the simulator needs an objectId: use the instance closest to where we saw it
    cands = [o for o in e.metadata["objects"] if o["objectType"] == target and o["pickupable"]]
    if not cands:
        return False, bool(boxes)
    obj = min(cands, key=lambda o: math.dist(est_pos, tuple(o["position"][k] for k in "xyz")))
    e = c.step(action="PickupObject", objectId=obj["objectId"])
    return e.metadata["lastActionSuccess"], bool(boxes)


def plot(grid, path, start, obj_pos, target, out):
    fig, ax = plt.subplots(figsize=(7, 7))
    xs, zs = zip(*[to_xz(c) for c in grid.free])
    ax.scatter(xs, zs, s=40, c="#dddddd", marker="s", label="Walkable floor")
    if path and len(path) > 1:
        px, pz = zip(*[to_xz(c) for c in path])
        ax.plot(px, pz, "-o", c="tab:blue", ms=3, label="Robot path")
    ax.scatter(*to_xz(start), s=150, c="tab:green", marker="^", label="Start")
    ax.scatter(obj_pos[0], obj_pos[2], s=250, c="tab:red", marker="*", label=target)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"A* path to {target}")
    plt.tight_layout()
    plt.savefig(out, dpi=150)


def main(cfg):
    random.seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    localizer = ObjectLocalizer(YOLO(cfg["model"]), cfg["size"], cfg["size"], cfg["fov"], cfg["conf"])
    c = Controller(scene=cfg["scene"], width=cfg["size"], height=cfg["size"],
                   fieldOfView=cfg["fov"], renderDepthImage=True, gridSize=0.25)
    c.step(action="InitialRandomSpawn", randomSeed=cfg["seed"], forceVisible=True)
    positions = c.step(action="GetReachablePositions").metadata["actionReturn"]
    grid = GridMap(positions)

    # 1) explore and find the target in the object map
    obj_map = explore(c, localizer, positions, cfg["n_views"])
    found = [o for o in obj_map.confirmed() if o["cls"] == cfg["target"]]
    if not found:
        print(f"{cfg['target']} was not found in the object map.")
        c.stop()
        return
    target = max(found, key=lambda o: o["sightings"])
    tx, _, tz = target["position"]
    print(f"Target {cfg['target']} at ({tx:.2f}, {tz:.2f}), seen {target['sightings']}x")
    for o in c.last_event.metadata["objects"]:
        if o["objectType"] == cfg["target"]:
            print(f"  (real {cfg['target']} is at ({o['position']['x']:.2f}, {o['position']['z']:.2f}))")

    # 2) start far from the target, facing a grid direction
    far = [p for p in positions if math.dist((p["x"], p["z"]), (tx, tz)) >= cfg["min_start_dist"]]
    start_pos = random.choice(far or positions)
    c.step(action="TeleportFull", position=start_pos,
           rotation=dict(x=0, y=random.choice([0, 90, 180, 270]), z=0), horizon=0, standing=True)
    start = to_cell(start_pos["x"], start_pos["z"])

    # 3) try the reachable cells closest to the target first (recovery if one fails)
    candidates = sorted(grid.goal_cells((tx, tz), cfg["reach"]),
                        key=lambda cell: math.dist(to_xz(cell), (tx, tz)))
    full_path, ok = [start], False
    for attempt, goal in enumerate(candidates[:cfg["max_attempts"]], 1):
        path = astar(grid, agent_cell(c), {goal})
        if path is None:
            print(f"Attempt {attempt}: no path to {to_xz(goal)}")
            continue
        actions, _ = path_to_actions(path, reset_view(c))
        for a in actions:
            if not c.step(action=a).metadata["lastActionSuccess"]:
                print(f"  {a} failed: {c.last_event.metadata['errorMessage'][:80]}")
                break
        full_path += path[1:]
        face(c, target["position"])
        Image.fromarray(c.last_event.frame).save(os.path.join(cfg["out_dir"], f"before_pickup_{attempt}.jpg"))
        ok, seen = pick_up(c, localizer, cfg["target"], target["position"])
        gx, gz = to_xz(goal)
        print(f"Attempt {attempt}: stand at ({gx:.2f}, {gz:.2f}), {len(path) - 1} steps, "
              f"YOLO saw it: {seen}, pickup: {'OK' if ok else 'failed'}")
        if ok:
            break

    plot(grid, full_path, start, target["position"], cfg["target"], os.path.join(cfg["out_dir"], "path.png"))
    Image.fromarray(c.last_event.frame).save(os.path.join(cfg["out_dir"], "after_pickup.jpg"))
    print("PICKUP SUCCESS" if ok else "PICKUP FAILED after all attempts")
    c.stop()


if __name__ == "__main__":
    main(CONFIG)
