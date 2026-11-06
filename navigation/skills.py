"""RoboBrain skills: explore, find, go to, look at, pick up, switch on/off, put."""
import os, sys, math, random
from PIL import Image
from ai2thor.controller import Controller
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "localization"))
from localizer import ObjectLocalizer, ObjectMap
from planner import GridMap, astar, path_to_actions, to_cell, to_xz

DEFAULTS = {
    "scene": "FloorPlan5",
    "model": os.path.join(HERE, "..", "object-detection", "models", "best_v3.pt"),
    "size": 600, "fov": 90,
    "n_views": 60,          # viewpoints while exploring
    "seed": 42,
    "conf": 0.5,            # YOLO threshold while exploring
    "verify_conf": 0.25,    # lower threshold when confirming an object we expect to see
    "reach": 1.3,           # max distance (m) to interact with an object
    "max_attempts": 6,      # approach cells to try before giving up
    "out_dir": os.path.join(HERE, "output"),
}


class Robot:
    """A simulated home robot. Each public method is one skill."""

    def __init__(self, **overrides):
        self.cfg = cfg = {**DEFAULTS, **overrides}
        random.seed(cfg["seed"])
        os.makedirs(cfg["out_dir"], exist_ok=True)
        self.loc = ObjectLocalizer(YOLO(cfg["model"]), cfg["size"], cfg["size"], cfg["fov"], cfg["conf"])
        self.c = Controller(scene=cfg["scene"], width=cfg["size"], height=cfg["size"],
                            fieldOfView=cfg["fov"], renderDepthImage=True, gridSize=0.25)
        self.c.step(action="InitialRandomSpawn", randomSeed=cfg["seed"], forceVisible=True)
        self.positions = self.c.step(action="GetReachablePositions").metadata["actionReturn"]
        self.grid = GridMap(self.positions)
        self.map = ObjectMap(merge_dist=0.3, min_sightings=2)
        self.home = None       # where the user is (the robot's starting cell)
        self.last_box = None   # last YOLO box seen by verify(), normalized 0-1

    # ---------- basics ----------
    @property
    def meta(self):
        return self.c.last_event.metadata

    def cell(self):
        p = self.meta["agent"]["position"]
        return to_cell(p["x"], p["z"])

    def snapshot(self, name):
        Image.fromarray(self.c.last_event.frame).save(os.path.join(self.cfg["out_dir"], name))

    def close(self):
        self.c.stop()

    # ---------- perception ----------
    def explore(self):
        """Look around from random viewpoints and build the object map."""
        print("Exploring...")
        for _ in range(self.cfg["n_views"]):
            e = self.c.step(action="TeleportFull", position=random.choice(self.positions),
                            rotation=dict(x=0, y=random.choice(range(0, 360, 30)), z=0),
                            horizon=30, standing=True)
            if e.metadata["lastActionSuccess"]:
                for d in self.loc.localize(e)[0]:
                    self.map.add(d)
        start = random.choice(self.positions)
        self.c.step(action="TeleportFull", position=start,
                    rotation=dict(x=0, y=random.choice([0, 90, 180, 270]), z=0), horizon=0, standing=True)
        self.home = self.cell()
        print("Found:", ", ".join(sorted({o["cls"] for o in self.map.confirmed()})))

    def find(self, cls):
        """Best map entry for a class (most sightings), or None."""
        hits = [o for o in self.map.confirmed() if o["cls"] == cls]
        return max(hits, key=lambda o: o["sightings"]) if hits else None

    def verify(self, cls, est_pos):
        """Re-detect the object in the current view; return (seen, refined position)."""
        e = self.c.last_event
        r = self.loc.model(e.frame[:, :, ::-1].copy(), conf=self.cfg["verify_conf"], verbose=False)[0]
        boxes = [b for b in r.boxes if self.loc.model.names[int(b.cls)] == cls]
        self.last_box = None
        if not boxes:
            return False, est_pos
        b = max(boxes, key=lambda b: float(b.conf))
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        s = self.cfg["size"]
        self.last_box = (x1 / s, y1 / s, x2 / s, y2 / s)
        d = self.loc.box_depth(e.depth_frame, x1, y1, x2, y2)
        if d:
            est_pos = self.loc.pixel_to_world((x1 + x2) / 2, (y1 + y2) / 2, d, e.metadata)
        return True, est_pos

    # ---------- motion ----------
    def reset_view(self):
        """Snap the heading to the grid and look straight ahead before driving."""
        a = self.meta["agent"]
        heading = round(a["rotation"]["y"] / 90) * 90 % 360
        self.c.step(action="TeleportFull", position=a["position"],
                    rotation=dict(x=0, y=heading, z=0), horizon=0, standing=True)
        return heading

    def go(self, goal):
        """Drive to a grid cell with A*. Returns the number of steps, or None."""
        path = astar(self.grid, self.cell(), {goal})
        if path is None:
            return None
        actions, _ = path_to_actions(path, self.reset_view())
        for a in actions:
            if not self.c.step(action=a).metadata["lastActionSuccess"]:
                print(f"  {a} failed: {self.meta['errorMessage'][:80]}")
                break
        return len(path) - 1

    def look_at(self, pos):
        """Turn and tilt the camera toward a 3D point, in place."""
        a = self.meta["agent"]["position"]
        dx, dz = pos[0] - a["x"], pos[2] - a["z"]
        yaw = math.degrees(math.atan2(dx, dz)) % 360
        pitch = math.degrees(math.atan2(self.meta["cameraPosition"]["y"] - pos[1], math.hypot(dx, dz)))
        pitch = max(-30, min(60, pitch))   # camera can only tilt between -30 and 60 degrees
        self.c.step(action="TeleportFull", position=a, rotation=dict(x=0, y=yaw, z=0),
                    horizon=pitch, standing=True)

    # ---------- skills ----------
    def interact(self, cls, action, prop):
        """Go next to an object and run a simulator action on it, with recovery."""
        target = self.find(cls)
        if not target:
            print(f"{cls} was not found in the map.")
            return False
        tx, _, tz = target["position"]
        cells = sorted(self.grid.goal_cells((tx, tz), self.cfg["reach"]),
                       key=lambda c: math.dist(to_xz(c), (tx, tz)))
        for i, goal in enumerate(cells[:self.cfg["max_attempts"]], 1):
            steps = self.go(goal)
            if steps is None:
                continue
            self.look_at(target["position"])
            seen, pos = self.verify(cls, target["position"])
            cands = [o for o in self.meta["objects"] if o["objectType"] == cls and o.get(prop)]
            ok = False
            if cands:
                obj = min(cands, key=lambda o: math.dist(pos, tuple(o["position"][k] for k in "xyz")))
                ok = self.c.step(action=action, objectId=obj["objectId"]).metadata["lastActionSuccess"]
            if not ok and action == "PutObject" and self.last_box:
                ok = self.put_at_box()
            print(f"  attempt {i}: {steps} steps, YOLO saw it: {seen}, {action}: {'OK' if ok else 'failed'}")
            if ok:
                self.snapshot(f"{action}_{cls}.jpg")
                return True
            print(f"    reason: {self.meta['errorMessage'][:90]}")
        return False

    def pick_up(self, cls):
        return self.interact(cls, "PickupObject", "pickupable")

    def switch_on(self, cls):
        return self.interact(cls, "ToggleObjectOn", "toggleable")

    def switch_off(self, cls):
        return self.interact(cls, "ToggleObjectOff", "toggleable")

    def holding(self):
        """Type of the object in the robot's hand, or None."""
        inv = self.meta["inventoryObjects"]
        return inv[0]["objectType"] if inv else None

    def put(self, receptacle):
        """Put the held object on or in a receptacle (sofa, table, ...)."""
        if not self.holding():
            print("Not holding anything.")
            return False
        return self.interact(receptacle, "PutObject", "receptacle")

    def put_at_box(self):
        """Fallback: aim at points inside the receptacle's YOLO box and put there."""
        x1, y1, x2, y2 = self.last_box
        for fy in (0.5, 0.65, 0.35, 0.8):
            for fx in (0.5, 0.3, 0.7):
                x, y = x1 + fx * (x2 - x1), y1 + fy * (y2 - y1)
                if self.c.step(action="PutObject", x=x, y=y).metadata["lastActionSuccess"]:
                    return True
        return False
