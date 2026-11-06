# Navigation & Pickup (A*)

The robot finds a target object, plans the shortest path to it, drives there, and picks it up.
Builds on the object map from [localization](../localization).

## Pipeline
1. **Explore** – the robot looks around from 40 random viewpoints and builds an object map (YOLO + depth, multi-frame fusion).
2. **Choose the target** – the map entry of the requested class with the most sightings.
3. **Grid map** – AI2-THOR reachable positions become a 0.25 m grid. Free cells are walkable; everything else (counters, tables, shelves) is an obstacle.
4. **Approach cells** – free cells within 1 m of the target, sorted from nearest to farthest.
5. **A\*** – shortest 4-connected path from the robot to the approach cell.
6. **Drive** – the path is converted into `RotateLeft` / `RotateRight` / `MoveAhead` actions.
7. **Face the object** – the robot turns and tilts its camera toward the target.
8. **Verify and pick up** – YOLO re-checks the target (lower confidence threshold, since we expect it there), then `PickupObject`.
9. **Recovery** – if the pickup fails (e.g. the object is hidden by a shelf), the robot plans to the next approach cell and tries again (up to 3 attempts).

## Example run (FloorPlan5, target: Apple)
| Step | Result |
|---|---|
| Apple in object map | seen 7x, position error 5 cm |
| Start distance | > 2 m from the target |
| Attempt 1 | 17 steps – pickup failed |
| Attempt 2 | 1 step – pickup failed |
| Attempt 3 | 2 steps – **pickup succeeded** |

![Path](output/path.png)

## Known limitations
- **Close-range verification:** at the approach cell, YOLO did not re-detect the apple (steep downward angle, object on a low shelf). The pickup used the position from the object map.
- **Scene randomness:** object placement can differ between runs even with the same seed.
- **Obstacles are assumed static.**

## Next steps
- Evaluate the success rate over many runs and different objects.
- Improve close-range verification (look-around scan, better camera angles).

## Files
- `planner.py` – `GridMap`, `astar`, `path_to_actions`
- `fetch.py` – full task: explore → plan → drive → pick up, with recovery
- `output/` – `path.png`, `before_pickup_*.jpg`, `after_pickup.jpg`

## Run
cd navigation
python fetch.py

## Update: 20-class model + camera tilt fix

### Camera tilt fix
The robot never tilted its camera down: `LookDown` with arbitrary degrees was not applied, so the tilt stayed at 0° and low objects were out of view. `face()` now sets heading and tilt directly, clamped to the allowed range (-30° to 60°). The plate that failed 6 attempts was then picked up on the first attempt.

| Before (tilt 0°) | After (tilt 58°) |
|---|---|
| ![](images/before_fix.jpg) | ![](images/after_fix.jpg) |

### Tests with the v2 model (after the fix)

| Scene | Target | Localization error | Steps | YOLO confirmed | Result |
|---|---|---|---|---|---|
| FloorPlan5 (kitchen) | Plate | 2 cm | 17 | no | success, attempt 1 |
| FloorPlan203 (living room) | Laptop | 11 cm | 40 | yes | success, attempt 1 |
| FloorPlan201 (living room) | RemoteControl | map entry 3.6 m from the real remote | – | – | failed after 6 attempts |

The RemoteControl failure came from a false map entry (likely a book: the confusion matrix shows Book→RemoteControl 14%). The robot did not grab the wrong object.

### Known issues / next steps
- Close-range verification: steep top-down views (50–60°) are missing from the training data → add such images.
- False map entries for small objects → require 3 sightings for small classes; after failing all attempts, mark the entry as false and try the next candidate.
- Large rooms need more exploration (`n_views` = 80).

### Usage
python fetch.py <Target> [Scene]
