from ai2thor.controller import Controller
c = Controller(scene="FloorPlan1", width=300, height=300)
e = c.step(action="MoveAhead")
print("Success:", e.metadata["lastActionSuccess"])
c.stop()
