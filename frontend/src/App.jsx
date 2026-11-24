import { useEffect, useRef, useState } from "react";
import { loadScene, getStatus, getMap, sendChat, sendVoice, frameUrl } from "./api";
import "./App.css";

const ROOMS = [
  { id: "FloorPlan201", name: "غرفة القعدة" },
  { id: "FloorPlan1", name: "المطبخ" },
  { id: "FloorPlan5", name: "مطبخ ٢" },
];
const STEP_ICON = { pending: "○", running: "◐", done: "✓", failed: "✕" };
const ACTION_AR = { bring: "جيب", water: "جيب مي بـ", turn_on: "شغّل", turn_off: "طفّي" };
const SUGGESTIONS = ["شغّل التلفزيون", "جيبلي الكتاب", "turn off the TV"];

function statusOf(s) {
  if (!s) return { text: "ما في اتصال بالروبوت", tone: "off" };
  if (s.scene_status === "loading") return { text: "عم يستكشف الغرفة…", tone: "busy" };
  if (s.scene_status === "error") return { text: "فشل تحميل الغرفة", tone: "bad" };
  if (s.scene_status !== "ready") return { text: "اختر غرفة للبدء", tone: "off" };
  if (s.busy) return { text: "عم يشتغل", tone: "busy" };
  return { text: "جاهز", tone: "ok" };
}

/* Floor plan: objects the robot found + the robot itself (units: cm) */
function FloorMap({ map }) {
  if (!map) return <div className="map-empty">الخريطة بتطلع بعد ما الروبوت يستكشف الغرفة</div>;
  const pts = [...map.objects.map((o) => [o.position[0], o.position[2]]), [map.robot.x, map.robot.z]];
  const xs = pts.map((p) => p[0]), zs = pts.map((p) => p[1]);
  const pad = 0.6;
  const minX = Math.min(...xs) - pad, maxX = Math.max(...xs) + pad;
  const minZ = Math.min(...zs) - pad, maxZ = Math.max(...zs) + pad;
  const X = (x) => (x - minX) * 100;
  const Y = (z) => (maxZ - z) * 100;
  const rad = (map.robot.rotation * Math.PI) / 180;

  return (
    <svg className="map-svg" viewBox={`0 0 ${(maxX - minX) * 100} ${(maxZ - minZ) * 100}`}>
      {map.objects.map((o, i) => (
        <g key={i} transform={`translate(${X(o.position[0])} ${Y(o.position[2])})`}>
          <circle r="7" className="obj-dot" />
          <text y="-12" className="obj-label">{o.cls}</text>
        </g>
      ))}
      <g className="robot" transform={`translate(${X(map.robot.x)} ${Y(map.robot.z)})`}>
        <circle r="22" className="robot-halo" />
        <circle r="12" className="robot-body" />
        <line x1="0" y1="0" x2={Math.sin(rad) * 34} y2={-Math.cos(rad) * 34} className="robot-heading" />
      </g>
    </svg>
  );
}

export default function App() {
  const [status, setStatus] = useState(null);
  const [map, setMap] = useState(null);
  const [frame, setFrame] = useState("");
  const [room, setRoom] = useState(ROOMS[0].id);
  const [text, setText] = useState("");
  const [messages, setMessages] = useState([]); // {from, text, taskId?, voice?}
  const [thinking, setThinking] = useState(false);
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState("");
  const recorder = useRef(null);
  const chatEnd = useRef(null);

  const ready = status?.scene_status === "ready";
  const locked = !ready || status?.busy || thinking || recording;

  // Status every second
  useEffect(() => {
    const id = setInterval(() => {
      getStatus().then(setStatus).catch(() => setStatus(null));
    }, 1000);
    return () => clearInterval(id);
  }, []);

  // Map + camera while a room is ready
  useEffect(() => {
    if (!ready) return;
    const tick = () => {
      getMap().then(setMap).catch(() => {});
      setFrame(frameUrl());
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [ready]);

  useEffect(() => {chatEnd.current?.scrollIntoView({ behavior: "smooth" })}, [messages, thinking]);

  const start = () => {
    setError("");
    setMap(null);
    setMessages([]);
    loadScene(room).catch((e) => setError(e.message));
  };

  const ask = async (msg) => {
    if (!msg.trim() || locked) return;
    setText("");
    setError("");
    setMessages((m) => [...m, { from: "user", text: msg }]);
    setThinking(true);
    try {
      const task = await sendChat(msg);
      setMessages((m) => [...m, { from: "robot", text: task.reply, taskId: task.id }]);
    } catch (e) {
      setError(e.message);
    } finally {
      setThinking(false);
    }
  };

  const toggleMic = async () => {
    if (recording) return recorder.current.stop();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const chunks = [];
      const r = new MediaRecorder(stream);
      r.ondataavailable = (e) => chunks.push(e.data);
      r.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        setRecording(false);
        setThinking(true);
        try {
          const res = await sendVoice(new Blob(chunks, { type: "audio/webm" }));
          setMessages((m) => [...m,
            { from: "user", text: res.heard, voice: true },
            { from: "robot", text: res.reply, taskId: res.id }]);
        } catch (e) {
          setError(e.message);
        } finally {
          setThinking(false);
        }
      };
      r.start();
      recorder.current = r;
      setRecording(true);
    } catch {
      setError("المتصفح ما سمح باستخدام المايك. اسمح له من أيقونة القفل جنب العنوان.");
    }
  };

  const st = statusOf(status);

  return (
    <div className="app" dir="rtl">
      <header className="top">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <h1>RoboBrain</h1>
        </div>
        <div className="room-picker">
          <select value={room} onChange={(e) => setRoom(e.target.value)} disabled={status?.busy}>
            {ROOMS.map((r) => <option key={r.id} value={r.id}>{r.name}</option>)}
          </select>
          <button className="btn" onClick={start} disabled={status?.busy || !status}>حمّل الغرفة</button>
        </div>
        <div className={`chip chip-${st.tone}`}><span className="dot" />{st.text}</div>
      </header>

      <main className="layout">
        <section className="chat" aria-label="المحادثة">
          <div className="messages">
            {messages.length === 0 && (
              <div className="empty">
                <p>اطلب شي من الروبوت بالكتابة أو بالصوت.</p>
                <div className="suggestions">
                  {SUGGESTIONS.map((s) => (
                    <button key={s} className="suggest" dir="auto" onClick={() => ask(s)} disabled={locked}>{s}</button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((m, i) => {
              const steps = m.taskId && status?.task?.id === m.taskId ? status.task.steps : null;
              return (
                <div key={i} className={`bubble ${m.from}`} dir="auto">
                  {m.voice && <span className="voice-tag">🎤 </span>}
                  {m.text}
                  {steps?.length > 0 && (
                    <ol className="steps" dir="rtl">
                      {steps.map((s, j) => (
                        <li key={j} className={`step ${s.status}`}>
                          <span className="step-icon">{STEP_ICON[s.status]}</span>
                          {ACTION_AR[s.action] ?? s.action} <bdi>{s.object}</bdi>
                        </li>
                      ))}
                    </ol>
                  )}
                </div>
              );
            })}
            {thinking && <div className="bubble robot thinking">عم يفكّر<span className="dots" /></div>}
            <div ref={chatEnd} />
          </div>

          {error && <p className="error" role="alert">{error}</p>}

          <div className="composer">
            <input
              dir="auto"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && ask(text)}
              placeholder={ready ? "شو بدك من الروبوت؟" : "حمّل غرفة أولاً"}
              disabled={locked}
            />
            <button className={`mic ${recording ? "on" : ""}`} onClick={toggleMic}
              disabled={!ready || status?.busy || thinking}
              aria-label={recording ? "وقّف التسجيل" : "سجّل صوت"}>
              {recording ? "■" : "🎤"}
            </button>
            <button className="btn send" onClick={() => ask(text)} disabled={locked || !text.trim()}>ابعت</button>
          </div>
        </section>

        <section className="world" aria-label="الغرفة">
          <div className="map-wrap">
            <FloorMap map={map} />
            {ready && frame && (
              <figure className="camera">
                <img src={frame} alt="كاميرا الروبوت" />
                <figcaption>كاميرا الروبوت</figcaption>
              </figure>
            )}
          </div>
          {status?.objects?.length > 0 && (
            <p className="found">
              لقى {status.objects.length} أغراض: <bdi>{status.objects.join("، ")}</bdi>
            </p>
          )}
        </section>
      </main>
    </div>
  );
}