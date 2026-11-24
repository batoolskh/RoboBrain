// All calls to the RoboBrain backend (FastAPI on Ubuntu) live here.
const BASE = import.meta.env.VITE_API_URL || "http://192.168.56.128:8000";

async function request(path, options = {}) {
  const res = await fetch(BASE + path, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Error ${res.status}`);
  }
  return res.json();
}

const post = (path, body) =>
  request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

// Room: start loading, then poll getStatus() until scene_status === "ready"
export const loadScene = (scene = "FloorPlan201") => post("/scene", { scene });

// Text request -> brain plan -> robot starts working
export const sendChat = (text) => post("/chat", { text });

// Voice request (audio Blob from the mic)
export function sendVoice(blob) {
  const form = new FormData();
  form.append("audio", blob, "voice.webm");
  return request("/voice", { method: "POST", body: form });
}

export const getStatus = () => request("/status"); // busy, scene_status, objects, task
export const getMap = () => request("/map");        // robot {x, z, rotation} + objects
export const getHealth = () => request("/health");

// Camera image URL (the ?t= forces a fresh image each time)
export const frameUrl = () => `${BASE}/frame?t=${Date.now()}`;