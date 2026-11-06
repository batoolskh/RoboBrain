"""Run one robot skill.

Usage:  python do.py <skill> <Object> [Scene]
Skills: pickup | on | off
Example: python do.py on Television FloorPlan201
"""
import sys
from skills import Robot

SKILLS = {"pickup": "pick_up", "on": "switch_on", "off": "switch_off"}

if len(sys.argv) < 3 or sys.argv[1] not in SKILLS:
    print(__doc__)
    sys.exit(1)

skill, target = sys.argv[1], sys.argv[2]
robot = Robot(**({"scene": sys.argv[3]} if len(sys.argv) > 3 else {}))
robot.explore()
ok = getattr(robot, SKILLS[skill])(target)
print("DONE" if ok else "FAILED")
robot.close()
