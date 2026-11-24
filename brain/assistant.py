"""RoboBrain assistant: user request -> LLM plan -> robot executes it in AI2-THOR.

Usage:  python assistant.py [Scene]      e.g.  python assistant.py FloorPlan5
"""
import math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.append(os.path.join(ROOT, "navigation"))

from brain import RobotBrain
brain = RobotBrain(os.path.join(HERE, "object_capabilities.json"))

os.chdir(os.path.join(ROOT, "navigation"))   # skills.py uses paths relative to navigation/
from skills import Robot
from planner import to_xz

SURFACES = ["DiningTable", "CoffeeTable", "CounterTop"]   # where we leave things for the user


# ---------- small helpers ----------

def step(robot, action, **kw):
    """Low-level action on a specific object (used when the robot is already standing there)."""
    return robot.c.step(action=action, forceAction=True, **kw).metadata["lastActionSuccess"]


def nearest(robot, obj_type, to=None):
    objs = [o for o in robot.meta["objects"] if o["objectType"] == obj_type]
    if not objs:
        return None
    if to is None:
        return objs[0]
    return min(objs, key=lambda o: math.dist(
        (o["position"]["x"], o["position"]["z"]), (to["x"], to["z"])))


# ---------- delivering ----------

def delivery_spot(robot):
    """The table/counter closest to the user (the robot's starting point)."""
    hx, hz = to_xz(robot.home)
    surfaces = [o for o in robot.map.confirmed() if o["cls"] in SURFACES]
    if not surfaces:
        return None
    best = min(surfaces, key=lambda o: (o["position"][0] - hx) ** 2 + (o["position"][2] - hz) ** 2)
    return best["cls"]


def deliver(robot):
    """Put the held object on the table nearest to the user; if that fails, hand it over."""
    spot = delivery_spot(robot)
    if spot and robot.put(spot):
        print(f"  placed it on the {spot} next to the user")
        return True
    return robot.return_home()


def drop(robot):
    """Fallback: free the hand by handing the held object to the user (simulated)."""
    inv = robot.meta["inventoryObjects"]
    if not inv:
        return True
    obj_id, obj_type = inv[0]["objectId"], inv[0]["objectType"]
    robot.c.step(action="DropHandObject", forceAction=True)
    ok = step(robot, "DisableObject", objectId=obj_id)
    print(f"  handed the {obj_type} to the user: {'OK' if ok else 'FAILED'}")
    return ok


# ---------- water ----------

def clear_sink_and_fill(robot, obj):
    """Robot is at the sink holding the container: empty the sink onto the counter, then fill."""
    sink = nearest(robot, "SinkBasin")
    if sink is None:
        return False
    faucet = nearest(robot, "Faucet", to=sink["position"])
    counter = nearest(robot, "CounterTop", to=sink["position"])
    mug_id = robot.meta["inventoryObjects"][0]["objectId"]
    inside = list(sink.get("receptacleObjectIds") or [])
    if not inside or faucet is None or counter is None:
        return False

    # 1) free the hand: leave the container on the counter for a moment
    if not step(robot, "PutObject", objectId=counter["objectId"]):
        return False
    # 2) move everything out of the sink onto the counter
    for item_id in inside:
        if step(robot, "PickupObject", objectId=item_id):
            if not step(robot, "PutObject", objectId=counter["objectId"]):
                step(robot, "DropHandObject")
            print(f"  moved the {item_id.split('|')[0]} out of the sink")
    # 3) container into the empty sink, run the water, take it back
    step(robot, "PickupObject", objectId=mug_id)
    step(robot, "PutObject", objectId=sink["objectId"])
    step(robot, "ToggleObjectOn", objectId=faucet["objectId"])
    for _ in range(10):                    # let the water run for a moment
        robot.c.step(action="Done")

    step(robot, "ToggleObjectOff", objectId=faucet["objectId"])
    filled = next(o for o in robot.meta["objects"] if o["objectId"] == mug_id)["isFilledWithLiquid"]
    print(f"  {obj} filled after clearing the sink: {filled}")
    return step(robot, "PickupObject", objectId=mug_id) and filled


def fill(robot, obj):
    """Fill the container with water at the sink, clearing the sink if needed."""
    if robot.fill_water(obj):
        return True
    if robot.holding() != obj:
        return False
    if clear_sink_and_fill(robot, obj):
        return True
    inv = robot.meta["inventoryObjects"]
    if not inv:
        return False
    ok = step(robot, "FillObjectWithLiquid", objectId=inv[0]["objectId"], fillLiquid="water")
    print(f"  last resort, filled the {obj} directly: {ok}")
    return ok


# ---------- brain action -> robot skills ----------

def run(robot, action, obj):
    """Translate one brain action into robot skills."""
    if action in ("bring", "water") and robot.holding():
        if not drop(robot):
            return False
    if action == "bring":
        return robot.pick_up(obj) and deliver(robot)
    if action == "water":
        return robot.pick_up(obj) and fill(robot, obj) and deliver(robot)
    if action == "turn_on":
        return robot.switch_on(obj)
    if action == "turn_off":
        return robot.switch_off(obj)
    return False


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else None
    robot = Robot(**({"scene": scene} if scene else {}))
    print("Exploring the house...")
    robot.explore()
    available = sorted({o["cls"] for o in robot.map.confirmed()})
    print("Ready! Type a request (empty line to quit).")
    try:
        while True:
            text = input("\nYou: ").strip()
            if not text:
                break
            plan = brain.ask(text, available)
            print("Robot:", plan["reply"])
            for s in plan["actions"]:
                a, o = s["action"], s["object"]
                print(f"  -> {a}({o}) ...", end=" ", flush=True)
                ok = run(robot, a, o)
                print("done" if ok else "failed")
                if not ok:
                    break
    finally:
        robot.close()


if __name__ == "__main__":
    main()
