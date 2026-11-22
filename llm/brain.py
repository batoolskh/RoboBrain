import json, re, requests

OLLAMA = "http://192.168.56.1:11434/api/chat"
MODEL = "qwen2.5:3b"
SKILLS = ["fetch", "fill_water", "open", "close", "turn_on", "turn_off"]

HOUSE = {
    "objects": ["Mug", "Cup", "Apple", "Bread", "Pillow", "Sink", "Fridge", "Window"],
    "fridge": ["Tomato", "Egg", "Lettuce", "Bottle"],
}
ALL = set(HOUSE["objects"] + HOUSE["fridge"])

SYSTEM = """You are RoboBrain, a home robot that helps a person at home.
Objects in the house:
- Food: Apple, Bread (on the counter); Tomato, Egg, Lettuce (inside Fridge)
- Drinks: Bottle (inside Fridge), or water from the Sink into a Mug or Cup
- Comfort: Pillow
- Places: Sink, Fridge, Window
Skills: fetch, fill_water, open, close, turn_on, turn_off.
Rules:
1. Write "reply" in the SAME language as the user. Arabic user -> Arabic reply only. Never use Chinese.
2. "reply" is 1-2 short friendly sentences.
3. Things inside the Fridge need opening the Fridge first.
4. For health complaints give a simple tip (rest, water). Never name medicines.
5. Output ONLY JSON."""

EXAMPLES = [
    ("بدي اشرب شي بارد", {"reply": "تكرم! رح افتح البراد وجيبلك قنينة مي باردة.", "intent": "drink",
        "plan": [{"skill": "open", "object": "Fridge"}, {"skill": "fetch", "object": "Bottle"}]}),
    ("الغرفة خانقة", {"reply": "معك حق، رح افتح الشباك ليفوت هوا نضيف.", "intent": "fresh_air",
        "plan": [{"skill": "open", "object": "Window"}]}),
    ("I'm so tired", {"reply": "Take a short rest. I'll bring you a pillow.", "intent": "rest",
        "plan": [{"skill": "fetch", "object": "Pillow"}]}),
]

SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "intent": {"type": "string"},
        "plan": {"type": "array", "items": {"type": "object", "properties": {
            "skill": {"type": "string", "enum": SKILLS},
            "object": {"type": "string", "enum": sorted(ALL)}},
            "required": ["skill", "object"]}},
    },
    "required": ["reply", "intent", "plan"],
}

def build(text):
    msgs = [{"role": "system", "content": SYSTEM}]
    for u, a in EXAMPLES:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": json.dumps(a, ensure_ascii=False)})
    msgs.append({"role": "user", "content": text})
    return msgs

def ask(text):
    r = requests.post(OLLAMA, json={
        "model": MODEL, "stream": False, "format": SCHEMA,
        "options": {"temperature": 0.2, "num_predict": 600}, "keep_alive": "30m",
        "messages": build(text),
    }, timeout=600)
    txt = r.json()["message"]["content"]
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return {"reply": txt[:200], "intent": "error", "plan": []}

def validate(out):
    out["reply"] = re.sub(r"[\u3000-\u9fff\uff00-\uffef]", "", str(out.get("reply", ""))).strip()
    plan = out.get("plan", [])
    good = [s for s in plan if s.get("skill") in SKILLS and s.get("object") in ALL]
    out["dropped"] = len(plan) - len(good)
    out["plan"] = good
    return out

TESTS = ["راسي عم يوجعني", "أنا عطشانة", "بردانة", "بدي شي آكله", "I'm thirsty"]

if __name__ == "__main__":
    results = []
    for t in TESTS:
        out = validate(ask(t))
        print("\n>>", t)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        results.append({"input": t, **out})
    with open("brain_test.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nSaved brain_test.json")
