"""Run a robot skill.

Usage:
  python do.py pickup <Object> [Scene]
  python do.py on|off <Object> [Scene]
  python do.py move <Object> <Receptacle> [Scene]
  python do.py bring <Object> [Scene]
  python do.py water <Container> [Scene]

Examples:
  python do.py on Television FloorPlan201
  python do.py move Laptop Sofa FloorPlan203
  python do.py bring Apple FloorPlan5
  python do.py water Mug FloorPlan5
"""
import sys
from skills import Robot

NARGS = {"pickup": 1, "on": 1, "off": 1, "move": 2, "bring": 1, "water": 1}

args = sys.argv[1:]
if not args or args[0] not in NARGS or len(args) < 1 + NARGS[args[0]]:
    print(__doc__)
    sys.exit(1)

skill = args[0]
targets = args[1:1 + NARGS[skill]]
rest = args[1 + NARGS[skill]:]
t = targets[0]

robot = Robot(**({"scene": rest[0]} if rest else {}))
robot.explore()

if skill == "pickup":
    ok = robot.pick_up(t)
elif skill == "on":
    ok = robot.switch_on(t)
elif skill == "off":
    ok = robot.switch_off(t)
elif skill == "move":
    ok = robot.pick_up(t) and robot.put(targets[1])
elif skill == "bring":
    ok = robot.pick_up(t) and robot.return_home()
else:  # water: get a container, fill it at the sink, bring it back
    ok = robot.pick_up(t) and robot.fill_water(t) and robot.return_home()

print("DONE" if ok else "FAILED")
robot.close()
