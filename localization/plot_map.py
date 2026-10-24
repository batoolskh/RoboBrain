"""Draw a top-down map: walkable floor, real objects, and estimated objects."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("output/object_map.json") as f:
    data = json.load(f)

fig, ax = plt.subplots(figsize=(8, 8))

# walkable floor
xs, zs = zip(*data["reachable"])
ax.scatter(xs, zs, s=40, c="#dddddd", marker="s", label="Walkable floor")

# one color per object class
classes = sorted({g["cls"] for g in data["ground_truth"]} | {o["cls"] for o in data["objects"]})
color = {c: plt.cm.tab10.colors[i % 10] for i, c in enumerate(classes)}

# real positions (circles)
for g in data["ground_truth"]:
    x, _, z = g["position"]
    ax.scatter(x, z, s=120, facecolors="none", edgecolors=[color[g["cls"]]], linewidths=2)

# estimated positions (crosses) + error label
for o in data["objects"]:
    x, _, z = o["position"]
    ax.scatter(x, z, s=80, marker="x", c=[color[o["cls"]]], linewidths=2.5)
    err = o.get("error_m")
    label = f'{o["cls"]} ({err*100:.0f} cm)' if err is not None else o["cls"]
    ax.annotate(label, (x, z), textcoords="offset points", xytext=(6, 6), fontsize=9)

ax.scatter([], [], s=120, facecolors="none", edgecolors="k", label="Real position")
ax.scatter([], [], s=80, marker="x", c="k", label="Estimated position")
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
ax.set_title(f'Object map - {data["config"]["scene"]}')
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("output/object_map.png", dpi=150)
print("Saved output/object_map.png")
