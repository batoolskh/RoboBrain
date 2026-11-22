"""RoboBrain API: connects the LLM brain and the robot (AI2-THOR) to the frontend.

Run:  uvicorn main:app --host 0.0.0.0 --port 8000
Docs: http://<ubuntu-ip>:8000/docs
"""
import io, os, sys, threading, time, uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "brain"))
import assistant   # loads the brain + robot skills (and moves into navigation/)

brain, Robot, run_action = assistant.brain, assistant.Robot, assistant.run

app = FastAPI(title="RoboBrain API", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],     # let the React app call us
                   allow_methods=["*"], allow_headers=["*"])

robot = None
robot_lock = threading.Lock()     # the simulator can only do one thing at a time
state = {"scene": None, "busy": False, "task": None}


class ChatRequest(BaseModel):
    text: str


class SceneRequest(BaseModel):
    scene: str = "FloorPlan201"


def found_objects():
    return sorted({o["cls"] for o in robot.map.confirmed()}) if robot else []


def controller():
    return getattr(robot, "c", None) or getattr(robot, "controller", None)


def execute(task):
    """Run the plan step by step in the background."""
    try:
        with robot_lock:
            for step in task["steps"]:
                step["status"] = "running"
                ok = run_action(robot, step["action"], step["object"])
                step["status"] = "done" if ok else "failed"
                if not ok:
                    task["status"] = "failed"
                    return
            task["status"] = "done"
    except Exception as e:
        task["status"], task["error"] = "error", str(e)
    finally:
        task["finished"] = time.time()
        state["busy"] = False


@app.get("/health")
def health():
    return {"status": "ok", "robot_ready": robot is not None,
            "scene": state["scene"], "busy": state["busy"]}


@app.post("/scene")
def load_scene(req: SceneRequest):
    """Open a room and let the robot explore it (takes ~1 minute)."""
    global robot
    if state["busy"]:
        raise HTTPException(409, "Robot is busy")
    with robot_lock:
        if robot is not None:
            robot.close()
        robot = Robot(scene=req.scene)
        robot.explore()
        state["scene"] = req.scene
    return {"scene": req.scene, "objects": found_objects()}


@app.post("/chat")
def chat(req: ChatRequest):
    """User request -> brain plan -> robot starts working in the background."""
    if robot is None:
        raise HTTPException(400, "Load a scene first: POST /scene")
    if state["busy"]:
        raise HTTPException(409, "Robot is busy with another task")
    plan = brain.ask(req.text, found_objects())
    task = {"id": uuid.uuid4().hex[:8], "text": req.text, "reply": plan["reply"],
            "steps": [{"action": a["action"], "object": a["object"], "status": "pending"}
                      for a in plan["actions"]],
            "status": "running" if plan["actions"] else "done",
            "started": time.time()}
    state["task"] = task
    if plan["actions"]:
        state["busy"] = True
        threading.Thread(target=execute, args=(task,), daemon=True).start()
    return task


@app.get("/status")
def status():
    """What the robot is doing right now."""
    return {"busy": state["busy"], "scene": state["scene"], "task": state["task"]}


@app.get("/map")
def get_map():
    """Objects the robot found + where the robot is."""
    if robot is None:
        raise HTTPException(400, "Load a scene first: POST /scene")
    agent = robot.meta()["agent"]
    return {"robot": {"x": agent["position"]["x"], "z": agent["position"]["z"],
                      "rotation": agent["rotation"]["y"]},
            "objects": [{"cls": o["cls"], "position": list(o["position"]),
                         "seen": o["sightings"]} for o in robot.map.confirmed()]}


@app.get("/frame", response_class=Response)
def frame():
    """Live camera image (JPEG)."""
    c = controller()
    if c is None or c.last_event is None:
        raise HTTPException(400, "No camera yet: POST /scene first")
    buf = io.BytesIO()
    Image.fromarray(c.last_event.frame).save(buf, format="JPEG", quality=80)
    return Response(buf.getvalue(), media_type="image/jpeg")
