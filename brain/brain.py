"""RoboBrain: turns a user's request into robot actions using a local LLM (Ollama)."""
import json, re, urllib.request

OLLAMA_URL = "http://192.168.56.1:11434/api/chat"
MODEL = "qwen3.5:4b"

# action -> capability the object needs (from object_capabilities.json)
ACTIONS = {"bring": "pickup", "water": "pickup", "turn_on": "toggle", "turn_off": "toggle"}
WATER_CONTAINERS = ["Mug", "Cup"]   # what the robot can fill at the sink

# Few-shot examples: teach the style, the Syrian dialect, and honesty
EXAMPLES = [
    ("عطشانة",
     {"reply": "من عيوني، رح عبّيلك كاسة مي وجبلك ياها.",
      "actions": [{"action": "water", "object": "Mug"}]}),
    ("راسي عم يوجعني",
     {"reply": "سلامتك! للأسف ما عندي دوا، بس رح جبلك كاسة مي يمكن تريّحك.",
      "actions": [{"action": "water", "object": "Cup"}]}),
    ("بدي اتفرج عالتلفزيون",
     {"reply": "تكرم عينك، رح شغّلك التلفزيون وجبلك الريموت.",
      "actions": [{"action": "turn_on", "object": "Television"},
                  {"action": "bring", "object": "RemoteControl"}]}),
    ("I want to make a salad",
     {"reply": "Sure! I'll bring you a tomato, a knife and a bowl.",
      "actions": [{"action": "bring", "object": "Tomato"},
                  {"action": "bring", "object": "Knife"},
                  {"action": "bring", "object": "Bowl"}]}),
    ("جبلي الكنباية",
     {"reply": "الكنباية تقيلة كتير عليّ، ما بقدر شيلها. بدك جبلك مخدة؟",
      "actions": []}),
]


class RobotBrain:
    def __init__(self, capabilities_file="object_capabilities.json",
                 url=OLLAMA_URL, model=MODEL):
        with open(capabilities_file) as f:
            self.caps = json.load(f)
        self.url, self.model = url, model

    def allowed_actions(self, available=None):
        """action -> objects it can be done to (only objects present in the house)."""
        objs = available or list(self.caps)
        allowed = {a: sorted(o for o in objs if self.caps.get(o, {}).get(cap))
                   for a, cap in ACTIONS.items()}
        allowed["water"] = [o for o in allowed["water"] if o in WATER_CONTAINERS]
        return allowed

    def _schema(self, allowed):
        options = [{"type": "object",
                    "properties": {"action": {"type": "string", "enum": [a]},
                                   "object": {"type": "string", "enum": objs}},
                    "required": ["action", "object"]}
                   for a, objs in allowed.items() if objs]
        return {"type": "object",
                "properties": {"reply": {"type": "string"},
                               "actions": {"type": "array", "items": {"anyOf": options}}},
                "required": ["reply", "actions"]}

    def _system(self, allowed, arabic):
        lines = "\n".join(f"- {a}: {', '.join(o)}" for a, o in allowed.items() if o)
        lang = ("Reply in Syrian (Levantine) Arabic dialect, warm and short."
                if arabic else "Reply in English, warm and short.")
        return f"""You are RoboBrain, the brain of a home-assistant robot.
Understand what the user NEEDS, even if they don't name an object, and plan robot actions.

Actions:
- bring: pick the object up and bring it to the user
- water: fill the container with water at the sink and bring it
- turn_on / turn_off: switch a device on or off

Objects each action works on (use ONLY these):
{lines}

Rules:
- Never invent objects or actions.
- Put in "actions" ONLY what the user asked for or clearly needs. Optional
  suggestions go in the reply as a question, NOT in actions.
- If the user needs something the robot can't do or doesn't have (medicine,
  food that isn't listed, heavy furniture), say so kindly and offer help.
- Never say the task is already done; say what you are going to do.
- {lang}
- If nothing fits, return an empty actions list."""

    def ask(self, text, available=None):
        allowed = self.allowed_actions(available)
        arabic = bool(re.search(r"[\u0600-\u06FF]", text))
        messages = [{"role": "system", "content": self._system(allowed, arabic)}]
        for q, a in EXAMPLES:
            messages += [{"role": "user", "content": q},
                         {"role": "assistant", "content": json.dumps(a, ensure_ascii=False)}]
        messages.append({"role": "user", "content": text})

        body = {"model": self.model, "messages": messages, "stream": False,
                "think": False, "format": self._schema(allowed),
                "options": {"temperature": 0}}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=180).read())
        out = json.loads(resp["message"]["content"])

        # safety net: drop anything the robot can't actually do
        out["actions"] = [x for x in out.get("actions", [])
                          if x.get("object") in allowed.get(x.get("action"), [])]
        return out


if __name__ == "__main__":
    brain = RobotBrain()
    tests = ["عطشانة", "راسي عم يوجعني وعطشانة", "بدي اعمل سلطة", "شغلي التلفزيون",
             "جبلي اللابتوب", "بدي اقرا شي", "I'm thirsty",
             "turn off the TV and bring me my phone"]
    for t in tests:
        r = brain.ask(t)
        acts = ", ".join(f"{a['action']}({a['object']})" for a in r["actions"]) or "-"
        print(f"\n> {t}\n  reply:   {r['reply']}\n  actions: {acts}")
