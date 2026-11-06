"""Run a robot skill.

Usage:
  python do.py pickup <Object> [Scene]
  python do.py on|off <Object> [Scene]
  python do.py move <Object> <Receptacle> [Scene]

Examples:
  python do.py on Television FloorPlan201
  python do.py move Laptop Sofa FloorPlan203
"""
import sys
from skills import Robot

NARGS = {"pickup": 1, "on": 1, "off": 1, "move": 2}

args = sys.argv[1:]
if not args or args[0] not in NARGS or len(args) < 1 + NARGS[args[0]]:
    print(__doc__)
    sys.exit(1)

skill = args[0]
targets = args[1:1 + NARGS[skill]]
rest = args[1 + NARGS[skill]:]

robot = Robot(**({"scene": rest[0]} if rest else {}))
robot.explore()

if skill == "pickup":
    ok = robot.pick_up(targets[0])
elif skill == "on":
    ok = robot.switch_on(targets[0])
elif skill == "off":
    ok = robot.switch_off(targets[0])
else:  # move = pick up, then put down
    ok = robot.pick_up(targets[0]) and robot.put(targets[1])

print("DONE" if ok else "FAILED")
robot.close()
