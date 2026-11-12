"""RoboBrain assistant: user request -> LLM plan -> robot executes it in AI2-THOR.

Usage:  python assistant.py [Scene]      e.g.  python assistant.py FloorPlan201
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.append(os.path.join(ROOT, "navigation"))

from brain import RobotBrain
brain = RobotBrain(os.path.join(HERE, "object_capabilities.json"))

os.chdir(os.path.join(ROOT, "navigation"))   # skills.py uses paths relative to navigation/
from skills import Robot


def run(robot, action, obj):
    """Translate one brain action into robot skills."""
    if action == "bring":
        return robot.pick_up(obj) and robot.return_home()
    if action == "water":
        return robot.pick_up(obj) and robot.fill_water(obj) and robot.return_home()
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
    print("Ready! Type a request (empty line to quit).")
    try:
        while True:
            text = input("\nYou: ").strip()
            if not text:
                break
            plan = brain.ask(text)
            print("Robot:", plan["reply"])
            for step in plan["actions"]:
                a, o = step["action"], step["object"]
                print(f"  -> {a}({o}) ...", end=" ", flush=True)
                ok = run(robot, a, o)
                print("done" if ok else "failed")
                if not ok:
                    break
    finally:
        robot.close()


if __name__ == "__main__":
    main()
