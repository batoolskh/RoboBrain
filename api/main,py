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

app = FastAPI(title="RoboBrain API", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],     # let the React app call us
                   allow_methods=["*"], allow_headers=["*"])

robot = None
robot_lock = threading.Lock()     # the simulator can only do one thing at a time
state = {"scene": None, "scene_status": "none",   # none | loading | ready | error
         "busy": False, "task": None, "error": None}


class ChatRequest(BaseModel):
    text: str


class SceneRequest(BaseModel):
    scene: str = "FloorPlan201"


def found_objects():
    return sorted({o["cls"] for o in robot.map.confirmed()}) if robot else []


def controller():
    return getattr(robot, "c", None) or getattr(robot, "controller", None)


# ---------- background workers ----------

def load_worker(scene):
    """Open a room and explore it in the background."""
    global robot
    try:
        with robot_lock:
            if robot is not None:
                robot.close()
                robot = None
            r = Robot(scene=scene)
            r.explore()
            robot = r
        state["scene_status"] = "ready"
    except Exception as e:
        state["scene_status"], state["error"] = "error", str(e)
    finally:
        state["busy"] = False


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


# ---------- endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok", "scene": state["scene"],
            "scene_status": state["scene_status"], "busy": state["busy"]}


@app.post("/scene")
def load_scene(req: SceneRequest):
    """Start loading a room. Returns immediately; poll /status until scene_status is 'ready'."""
    if state["busy"]:
        raise HTTPException(409, "Robot is busy")
    state.update(scene=req.scene, scene_status="loading", busy=True, error=None, task=None)
    threading.Thread(target=load_worker, args=(req.scene,), daemon=True).start()
    return {"scene": req.scene, "scene_status": "loading"}


@app.post("/chat")
def chat(req: ChatRequest):
    """User request -> brain plan -> robot starts working in the background."""
    if state["scene_status"] != "ready":
        raise HTTPException(400, "No room ready yet: POST /scene and wait for 'ready'")
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
    """Everything the frontend needs to know right now."""
    return {"busy": state["busy"], "scene": state["scene"],
            "scene_status": state["scene_status"], "error": state["error"],
            "objects": found_objects() if state["scene_status"] == "ready" else [],
            "task": state["task"]}


@app.get("/map")
def get_map():
    """Objects the robot found + where the robot is."""
    if state["scene_status"] != "ready":
        raise HTTPException(400, "No room ready yet")
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
        raise HTTPException(400, "No camera yet")
    buf = io.BytesIO()
    Image.fromarray(c.last_event.frame).save(buf, format="JPEG", quality=80)
    return Response(buf.getvalue(), media_type="image/jpeg")
