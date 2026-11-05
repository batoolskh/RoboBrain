"""Object localization: YOLO detections + depth -> 3D world coordinates (AI2-THOR)."""
import math
import numpy as np


class ObjectLocalizer:
    """Turns YOLO detections into 3D world positions using the depth image."""

    def __init__(self, model, width, height, fov=90, conf=0.5, center_ratio=0.2):
        self.model = model
        self.w, self.h = width, height
        # focal length in pixels (AI2-THOR fieldOfView is vertical)
        self.f = (height / 2) / math.tan(math.radians(fov / 2))
        self.conf = conf
        self.center_ratio = center_ratio

    def pixel_to_world(self, u, v, depth, meta):
        """Pixel (u, v) + depth in meters -> world (x, y, z) using the camera pose."""
        cam = meta["cameraPosition"]
        pitch = math.radians(meta["agent"]["cameraHorizon"])
        yaw = math.radians(meta["agent"]["rotation"]["y"])
        # camera frame
        xc = (u - self.w / 2) * depth / self.f
        yc = -(v - self.h / 2) * depth / self.f
        zc = depth
        # undo camera tilt
        ya = yc * math.cos(pitch) - zc * math.sin(pitch)
        za = yc * math.sin(pitch) + zc * math.cos(pitch)
        # undo agent heading
        xw = xc * math.cos(yaw) + za * math.sin(yaw)
        zw = -xc * math.sin(yaw) + za * math.cos(yaw)
        return (cam["x"] + xw, cam["y"] + ya, cam["z"] + zw)

    def box_depth(self, depth_frame, x1, y1, x2, y2):
        """Median depth of a small patch at the box center (robust to edges)."""
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        hw, hh = (x2 - x1) * self.center_ratio / 2, (y2 - y1) * self.center_ratio / 2
        a1, a2 = max(0, int(cx - hw)), min(self.w, int(cx + hw) + 1)
        b1, b2 = max(0, int(cy - hh)), min(self.h, int(cy + hh) + 1)
        patch = depth_frame[b1:b2, a1:a2]
        return float(np.median(patch)) if patch.size else None

    def localize(self, event):
        """Detect objects in one frame and return their world positions."""
        result = self.model(event.frame[:, :, ::-1].copy(), conf=self.conf, verbose=False)[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            d = self.box_depth(event.depth_frame, x1, y1, x2, y2)
            if d is None or d <= 0:
                continue
            u, v = (x1 + x2) / 2, (y1 + y2) / 2
            detections.append({
                "cls": self.model.names[int(box.cls)],
                "conf": float(box.conf),
                "position": self.pixel_to_world(u, v, d, event.metadata),
            })
        return detections, result


class ObjectMap:
    """Fuses detections from many frames into one entry per real object."""

    def __init__(self, merge_dist=0.3, min_sightings=2):
        self.merge_dist = merge_dist        # detections closer than this (m) = same object
        self.min_sightings = min_sightings  # seen fewer times = probably a false detection
        self.objects = []

    def add(self, det):
        for obj in self.objects:
            if obj["cls"] == det["cls"] and math.dist(obj["position"], det["position"]) < self.merge_dist:
                n = obj["sightings"]
                obj["position"] = tuple((p * n + q) / (n + 1)
                                        for p, q in zip(obj["position"], det["position"]))
                obj["conf"] = max(obj["conf"], det["conf"])
                obj["sightings"] = n + 1
                return
        self.objects.append({**det, "sightings": 1})

    def confirmed(self):
        """Objects seen enough times to trust."""
        return [o for o in self.objects if o["sightings"] >= self.min_sightings]

    def rejected(self):
        return [o for o in self.objects if o["sightings"] < self.min_sightings]
