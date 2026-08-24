"""
Local web tool for manually reviewing/fixing Kradangnga's point
annotations. Visual review (this session) confirmed a real problem: many
points sit in gaps/shadow between tree crowns rather than centered on a
crown, inconsistently across boxes -- not a simple uniform coordinate
offset, so it needs a human eyeballing the actual imagery, not an
automated fix.

Serves one full Kradangnga source tile (8192x4283, as JPEG) at a time in
a pan/zoom viewer with every box's points drawn in their real position
plus box outlines, so problems are visible in spatial context rather than
one disconnected patch at a time. Click empty canopy to add a point,
click near an existing point to delete it, drag to nudge it -- new/moved
points are attributed to whichever annotated box contains that
coordinate. Save writes straight back to
Applicatno/annotations/Kradangnga_800_<N>.json, preserving the existing
schema (other fields untouched) so mask generation and tiling keep
working unchanged once this is done.

Usage:
    python segmentation/review_tool.py
    Then open http://127.0.0.1:5050 in a browser.
"""

import json
from pathlib import Path

import cv2
from flask import Flask, jsonify, request, Response

ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = ROOT / "Applicatno" / "annotations"
COCONUT_DIR = ROOT / "Coconut" / "Kradangnga Thailand"
PROGRESS_PATH = Path(__file__).resolve().parent / "kradangnga_review_progress.json"

TILES = [1, 2, 3, 4]
JPEG_QUALITY = 90

app = Flask(__name__)

_image_cache = {}


def load_image(tile_n: int):
    if tile_n not in _image_cache:
        path = COCONUT_DIR / f"Kradangnga_800_{tile_n}.png"
        _image_cache[tile_n] = cv2.imread(str(path))
    return _image_cache[tile_n]


def annotation_path(tile_n: int) -> Path:
    return ANNOTATIONS_DIR / f"Kradangnga_800_{tile_n}.json"


def load_annotations(tile_n: int):
    with open(annotation_path(tile_n), "r", encoding="utf-8") as f:
        return json.load(f)


def save_annotations(tile_n: int, data):
    with open(annotation_path(tile_n), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_progress():
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def box_containing(data, x, y):
    """Index of the box whose rect contains (x, y), or the nearest box if none does."""
    best_idx, best_dist = None, None
    for i, box in enumerate(data):
        c = box["Coordinates"]
        x1, y1 = c["Top-left"]
        x2, y2 = c["Bottom-right"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        d = (cx - x) ** 2 + (cy - y) ** 2
        if best_dist is None or d < best_dist:
            best_dist, best_idx = d, i
    return best_idx


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/api/tiles")
def api_tiles():
    progress = load_progress()
    result = []
    for tile_n in TILES:
        img = load_image(tile_n)
        result.append({
            "tile": tile_n,
            "width": img.shape[1],
            "height": img.shape[0],
            "reviewed": progress.get(str(tile_n), False),
        })
    return jsonify(result)


@app.route("/api/tile_data/<int:tile_n>")
def api_tile_data(tile_n):
    data = load_annotations(tile_n)
    boxes = []
    for i, box in enumerate(data):
        c = box["Coordinates"]
        boxes.append({
            "index": i,
            "box_id": box["Box ID"],
            "rect": [c["Top-left"][0], c["Top-left"][1], c["Bottom-right"][0], c["Bottom-right"][1]],
            "points": box.get("Points", []),
        })
    return jsonify({"boxes": boxes})


@app.route("/api/tile_image/<int:tile_n>")
def api_tile_image(tile_n):
    img = load_image(tile_n)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/api/tile_data/<int:tile_n>", methods=["POST"])
def api_save_tile(tile_n):
    payload = request.get_json()
    points_by_box = payload["points_by_box"]  # { "<box_index>": [[x,y],...], ... }
    mark_reviewed = payload.get("mark_reviewed", False)

    data = load_annotations(tile_n)
    for box_idx_str, pts in points_by_box.items():
        box_idx = int(box_idx_str)
        data[box_idx]["Points"] = [[round(px, 1), round(py, 1)] for px, py in pts]
    save_annotations(tile_n, data)

    if mark_reviewed:
        progress = load_progress()
        progress[str(tile_n)] = True
        save_progress(progress)

    return jsonify({"ok": True})


INDEX_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Kradangnga Annotation Review</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; height: 100vh; background: #1a1a1a; color: #eee; display: flex; flex-direction: column; overflow: hidden; }
  #toolbar { padding: 8px 12px; border-bottom: 1px solid #444; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  button, select { padding: 6px 12px; cursor: pointer; }
  #status { color: #9c9; margin-left: 8px; }
  #help { color: #999; font-size: 12px; padding: 4px 12px; }
  #viewport { flex: 1; overflow: hidden; position: relative; background: #000; cursor: grab; }
  #viewport.grabbing { cursor: grabbing; }
  #stage { position: absolute; top: 0; left: 0; transform-origin: 0 0; }
  #stage img { display: block; user-select: none; -webkit-user-drag: none; }
  svg { position: absolute; top: 0; left: 0; overflow: visible; }
  .box-rect { fill: none; stroke: rgba(0,150,255,0.5); stroke-width: 2; }
  .pt-ring { fill: none; stroke: red; stroke-width: 2; }
  .pt-dot { fill: yellow; }
  .tile-btn.done { color: #6f6; }
</style>
</head>
<body>
<div id="toolbar">
  <span>Tile:</span>
  <select id="tileSelect"></select>
  <button id="saveBtn">Save (S)</button>
  <button id="markReviewedBtn">Save &amp; Mark Reviewed</button>
  <button id="zoomInBtn">Zoom In (+)</button>
  <button id="zoomOutBtn">Zoom Out (-)</button>
  <button id="resetViewBtn">Reset View (0)</button>
  <span id="status"></span>
</div>
<div id="help">
  Scroll = zoom. Drag background = pan. Click empty canopy = add point. Click near a point = delete it. Drag a point = move it.
</div>
<div id="viewport">
  <div id="stage">
    <img id="tileImg" draggable="false">
    <svg id="overlay"></svg>
  </div>
</div>
<script>
let tiles = [];
let currentTile = null;
let boxes = [];           // [{index, box_id, rect:[x1,y1,x2,y2], points:[[x,y],...]}]
let scale = 0.15;
let panX = 0, panY = 0;
let dragging = null;      // {type:'point', boxIdx, ptIdx} or {type:'pan', startX, startY, startPanX, startPanY}
const HIT_RADIUS_SCREEN = 10;

const viewport = document.getElementById('viewport');
const stage = document.getElementById('stage');
const tileImg = document.getElementById('tileImg');
const overlay = document.getElementById('overlay');

async function loadTileList() {
  const res = await fetch('/api/tiles');
  tiles = await res.json();
  const sel = document.getElementById('tileSelect');
  sel.innerHTML = '';
  for (const t of tiles) {
    const opt = document.createElement('option');
    opt.value = t.tile;
    opt.textContent = `Kradangnga_800_${t.tile}` + (t.reviewed ? ' [reviewed]' : '');
    sel.appendChild(opt);
  }
  sel.onchange = () => selectTile(parseInt(sel.value));
  selectTile(tiles[0].tile);
}

async function selectTile(tileN) {
  currentTile = tileN;
  const res = await fetch(`/api/tile_data/${tileN}`);
  const data = await res.json();
  boxes = data.boxes;
  tileImg.src = `/api/tile_image/${tileN}`;
  tileImg.onload = () => {
    fitView();
    render();
  };
  document.getElementById('status').textContent =
    `Tile ${tileN} — ${boxes.length} boxes, ${boxes.reduce((s,b)=>s+b.points.length,0)} points`;
}

function fitView() {
  const vw = viewport.clientWidth, vh = viewport.clientHeight;
  scale = Math.min(vw / tileImg.naturalWidth, vh / tileImg.naturalHeight) * 0.95;
  panX = (vw - tileImg.naturalWidth * scale) / 2;
  panY = (vh - tileImg.naturalHeight * scale) / 2;
  applyTransform();
}

function applyTransform() {
  stage.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
}

function render() {
  overlay.setAttribute('width', tileImg.naturalWidth);
  overlay.setAttribute('height', tileImg.naturalHeight);
  overlay.setAttribute('viewBox', `0 0 ${tileImg.naturalWidth} ${tileImg.naturalHeight}`);
  let svg = '';
  const strokeW = Math.max(1, 2 / scale);
  const ringR = Math.max(3, 6 / scale);
  const dotR = Math.max(1, 1.5 / scale);
  for (const b of boxes) {
    const [x1, y1, x2, y2] = b.rect;
    svg += `<rect class="box-rect" x="${x1}" y="${y1}" width="${x2-x1}" height="${y2-y1}" style="stroke-width:${strokeW}"/>`;
    for (const [px, py] of b.points) {
      svg += `<circle class="pt-ring" cx="${px}" cy="${py}" r="${ringR}" style="stroke-width:${strokeW}"/>`;
      svg += `<circle class="pt-dot" cx="${px}" cy="${py}" r="${dotR}"/>`;
    }
  }
  overlay.innerHTML = svg;
}

function screenToImage(clientX, clientY) {
  const rect = viewport.getBoundingClientRect();
  const x = (clientX - rect.left - panX) / scale;
  const y = (clientY - rect.top - panY) / scale;
  return [x, y];
}

function findNearbyPoint(imgX, imgY) {
  const hitRadiusImg = HIT_RADIUS_SCREEN / scale;
  for (const b of boxes) {
    for (let i = 0; i < b.points.length; i++) {
      const [px, py] = b.points[i];
      if (Math.hypot(px - imgX, py - imgY) < hitRadiusImg) {
        return { boxIdx: b.index, ptIdx: i };
      }
    }
  }
  return null;
}

function boxIndexContaining(x, y) {
  let best = null, bestDist = Infinity;
  for (const b of boxes) {
    const [x1, y1, x2, y2] = b.rect;
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2) return b.index;
    const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
    const d = (cx - x) ** 2 + (cy - y) ** 2;
    if (d < bestDist) { bestDist = d; best = b.index; }
  }
  return best;
}

viewport.addEventListener('wheel', (e) => {
  e.preventDefault();
  const rect = viewport.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const imgXBefore = (mx - panX) / scale, imgYBefore = (my - panY) / scale;
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  scale = Math.min(Math.max(scale * factor, 0.02), 8);
  panX = mx - imgXBefore * scale;
  panY = my - imgYBefore * scale;
  applyTransform();
  render();
}, { passive: false });

viewport.addEventListener('mousedown', (e) => {
  const [imgX, imgY] = screenToImage(e.clientX, e.clientY);
  const hit = findNearbyPoint(imgX, imgY);
  if (hit) {
    dragging = { type: 'point', ...hit, moved: false };
  } else {
    dragging = { type: 'pan', startClientX: e.clientX, startClientY: e.clientY, startPanX: panX, startPanY: panY };
    viewport.classList.add('grabbing');
  }
});

viewport.addEventListener('mousemove', (e) => {
  if (!dragging) return;
  if (dragging.type === 'pan') {
    panX = dragging.startPanX + (e.clientX - dragging.startClientX);
    panY = dragging.startPanY + (e.clientY - dragging.startClientY);
    applyTransform();
  } else if (dragging.type === 'point') {
    const [imgX, imgY] = screenToImage(e.clientX, e.clientY);
    const box = boxes.find(b => b.index === dragging.boxIdx);
    box.points[dragging.ptIdx] = [imgX, imgY];
    dragging.moved = true;
    render();
  }
});

viewport.addEventListener('mouseup', (e) => {
  if (!dragging) return;
  if (dragging.type === 'point' && !dragging.moved) {
    // pure click on existing point = delete
    const box = boxes.find(b => b.index === dragging.boxIdx);
    box.points.splice(dragging.ptIdx, 1);
    render();
  } else if (dragging.type === 'pan') {
    const moved = Math.hypot(e.clientX - dragging.startClientX, e.clientY - dragging.startClientY);
    if (moved < 3) {
      // pure click on empty canopy = add point
      const [imgX, imgY] = screenToImage(e.clientX, e.clientY);
      const boxIdx = boxIndexContaining(imgX, imgY);
      if (boxIdx !== null) {
        const box = boxes.find(b => b.index === boxIdx);
        box.points.push([imgX, imgY]);
        render();
      }
    }
  }
  dragging = null;
  viewport.classList.remove('grabbing');
});

async function save(markReviewed) {
  const points_by_box = {};
  for (const b of boxes) points_by_box[b.index] = b.points;
  await fetch(`/api/tile_data/${currentTile}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ points_by_box, mark_reviewed: markReviewed }),
  });
  const total = boxes.reduce((s,b)=>s+b.points.length,0);
  document.getElementById('status').textContent = `Saved — ${total} points` + (markReviewed ? ' (marked reviewed)' : '');
  if (markReviewed) await loadTileList();
}

document.getElementById('saveBtn').onclick = () => save(false);
document.getElementById('markReviewedBtn').onclick = () => save(true);
document.getElementById('zoomInBtn').onclick = () => { scale = Math.min(scale * 1.3, 8); applyTransform(); render(); };
document.getElementById('zoomOutBtn').onclick = () => { scale = Math.max(scale / 1.3, 0.02); applyTransform(); render(); };
document.getElementById('resetViewBtn').onclick = fitView;
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT') return;
  if (e.key === 's' || e.key === 'S') save(false);
  if (e.key === '0') fitView();
  if (e.key === '+' || e.key === '=') { scale = Math.min(scale * 1.3, 8); applyTransform(); render(); }
  if (e.key === '-') { scale = Math.max(scale / 1.3, 0.02); applyTransform(); render(); }
});
window.addEventListener('resize', () => { if (tileImg.complete) render(); });

loadTileList();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print(f"Annotations dir: {ANNOTATIONS_DIR}")
    print(f"Images dir: {COCONUT_DIR}")
    print("Open http://127.0.0.1:5050 in your browser")
    app.run(host="127.0.0.1", port=5050, debug=False)
