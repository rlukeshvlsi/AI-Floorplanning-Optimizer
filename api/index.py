import base64
import io
import math
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "floorplan_model_v3.pkl"

CHIP_WIDTH = 200
CHIP_HEIGHT = 200

MACROS = {
    "A": {"w": 40, "h": 30},
    "B": {"w": 50, "h": 35},
    "C": {"w": 30, "h": 30},
    "D": {"w": 40, "h": 40},
}

NETS = [
    ("A", "B"),
    ("A", "C"),
    ("B", "D"),
    ("C", "D"),
]

NUM_CANDIDATES = 500
TOP_K = 200


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="AI Floorplanning Optimizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Random Forest model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# BASIC FLOORPLAN FUNCTIONS
# ============================================================

def clamp(value, low, high):
    return max(low, min(value, high))


def get_center(macro_name, placement):
    x = placement[macro_name]["x"]
    y = placement[macro_name]["y"]

    w = MACROS[macro_name]["w"]
    h = MACROS[macro_name]["h"]

    return (
        x + w / 2,
        y + h / 2,
    )


def connection_distances(placement):
    distances = {}

    for a, b in NETS:
        ax, ay = get_center(a, placement)
        bx, by = get_center(b, placement)

        distance = math.sqrt(
            (ax - bx) ** 2 +
            (ay - by) ** 2
        )

        distances[f"{a}{b}_distance"] = distance

    return distances


def create_features(placement):
    distances = connection_distances(placement)

    return {
        "Ax": placement["A"]["x"] / CHIP_WIDTH,
        "Ay": placement["A"]["y"] / CHIP_HEIGHT,

        "Bx": placement["B"]["x"] / CHIP_WIDTH,
        "By": placement["B"]["y"] / CHIP_HEIGHT,

        "Cx": placement["C"]["x"] / CHIP_WIDTH,
        "Cy": placement["C"]["y"] / CHIP_HEIGHT,

        "Dx": placement["D"]["x"] / CHIP_WIDTH,
        "Dy": placement["D"]["y"] / CHIP_HEIGHT,

        "AB_distance": distances["AB_distance"],
        "AC_distance": distances["AC_distance"],
        "BD_distance": distances["BD_distance"],
        "CD_distance": distances["CD_distance"],
    }


# ============================================================
# ACTUAL QOR
# ============================================================

def calculate_hpwl(placement):

    xs = []
    ys = []

    for name in MACROS:

        x = placement[name]["x"]
        y = placement[name]["y"]

        w = MACROS[name]["w"]
        h = MACROS[name]["h"]

        xs.extend([x, x + w])
        ys.extend([y, y + h])

    hpwl = (
        max(xs) - min(xs)
        +
        max(ys) - min(ys)
    )

    return hpwl


def calculate_overlap_area(placement):

    names = list(MACROS.keys())

    total_overlap = 0

    for i in range(len(names)):

        for j in range(i + 1, len(names)):

            a = names[i]
            b = names[j]

            ax1 = placement[a]["x"]
            ay1 = placement[a]["y"]

            ax2 = ax1 + MACROS[a]["w"]
            ay2 = ay1 + MACROS[a]["h"]

            bx1 = placement[b]["x"]
            by1 = placement[b]["y"]

            bx2 = bx1 + MACROS[b]["w"]
            by2 = by1 + MACROS[b]["h"]

            overlap_x = max(
                0,
                min(ax2, bx2) - max(ax1, bx1)
            )

            overlap_y = max(
                0,
                min(ay2, by2) - max(ay1, by1)
            )

            total_overlap += overlap_x * overlap_y

    return total_overlap


def boundary_violation(placement):

    violation = 0

    for name in MACROS:

        x = placement[name]["x"]
        y = placement[name]["y"]

        w = MACROS[name]["w"]
        h = MACROS[name]["h"]

        if x < 0:
            violation += abs(x)

        if y < 0:
            violation += abs(y)

        if x + w > CHIP_WIDTH:
            violation += x + w - CHIP_WIDTH

        if y + h > CHIP_HEIGHT:
            violation += y + h - CHIP_HEIGHT

    return violation


def actual_qor(placement):

    hpwl = calculate_hpwl(placement)
    overlap = calculate_overlap_area(placement)
    boundary = boundary_violation(placement)

    score = 100

    score -= hpwl * 0.05
    score -= overlap * 0.10
    score -= boundary * 2

    return max(0, round(score, 4))


# ============================================================
# IMAGE DETECTION
# ============================================================

def detect_macros_from_image(image):

    image = image.convert("RGB")

    # Reduce image size for faster processing
    max_size = 600

    scale = min(
        1.0,
        max_size / max(image.width, image.height)
    )

    small = image.resize(
        (
            max(1, int(image.width * scale)),
            max(1, int(image.height * scale)),
        )
    )

    arr = np.asarray(small)

    r = arr[:, :, 0].astype(int)
    g = arr[:, :, 1].astype(int)
    b = arr[:, :, 2].astype(int)

    # Detect the light-blue macro rectangles
    mask = (
        (r >= 100)
        & (r <= 210)
        & (g >= 145)
        & (g <= 225)
        & (b >= 175)
        & (b <= 245)
        & (g > r)
        & (b > g)
    )

    ys, xs = np.where(mask)

    if len(xs) < 1000:
        raise ValueError(
            "Could not detect the blue macro blocks. "
            "Please upload a floorplan image generated by the project."
        )

    # --------------------------------------------------------
    # K-means clustering into four macro regions
    # --------------------------------------------------------

    points = np.column_stack((xs, ys)).astype(float)

    # Sample points if image is large
    if len(points) > 15000:

        rng = np.random.default_rng(42)

        indices = rng.choice(
            len(points),
            size=15000,
            replace=False
        )

        points = points[indices]

    # Initial centers based on spatial quantiles
    centers = np.array([
        points[np.argmin(points[:, 0] + points[:, 1])],
        points[np.argmax(points[:, 0] - points[:, 1])],
        points[np.argmax(points[:, 0] + points[:, 1])],
        points[np.argmax(points[:, 1] - points[:, 0])],
    ])

    # Remove duplicate initial centers
    unique_centers = []

    for center in centers:

        if not any(
            np.linalg.norm(center - c) < 10
            for c in unique_centers
        ):
            unique_centers.append(center)

    while len(unique_centers) < 4:

        unique_centers.append(
            points[
                random.randint(
                    0,
                    len(points) - 1
                )
            ]
        )

    centers = np.array(unique_centers[:4])

    # K-means
    for _ in range(20):

        distances = np.linalg.norm(
            points[:, None, :] - centers[None, :, :],
            axis=2
        )

        labels = np.argmin(
            distances,
            axis=1
        )

        new_centers = []

        for k in range(4):

            cluster = points[labels == k]

            if len(cluster) == 0:
                new_centers.append(centers[k])
            else:
                new_centers.append(
                    cluster.mean(axis=0)
                )

        new_centers = np.array(new_centers)

        if np.allclose(
            centers,
            new_centers,
            atol=1
        ):
            break

        centers = new_centers

    # --------------------------------------------------------
    # Get bounding boxes
    # --------------------------------------------------------

    boxes = []

    for k in range(4):

        cluster = points[labels == k]

        if len(cluster) < 100:
            continue

        x1 = cluster[:, 0].min()
        x2 = cluster[:, 0].max()

        y1 = cluster[:, 1].min()
        y2 = cluster[:, 1].max()

        boxes.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "area": (x2 - x1) * (y2 - y1),
        })

    if len(boxes) != 4:
        raise ValueError(
            f"Detected {len(boxes)} macro regions instead of 4."
        )

    # --------------------------------------------------------
    # Convert image coordinates to 200x200 chip coordinates
    # --------------------------------------------------------

    W, H = small.size

    # Approximate plotting area used by the project's
    # Matplotlib floorplan images.
    plot_left = 0.052 * W
    plot_right = 0.968 * W
    plot_top = 0.024 * H
    plot_bottom = 0.931 * H

    placements = []

    for box in boxes:

        px = box["x1"]
        py = box["y2"]

        x = (
            (px - plot_left)
            /
            (plot_right - plot_left)
            *
            CHIP_WIDTH
        )

        y = (
            (plot_bottom - py)
            /
            (plot_bottom - plot_top)
            *
            CHIP_HEIGHT
        )

        placements.append({
            "x": float(clamp(x, 0, CHIP_WIDTH)),
            "y": float(clamp(y, 0, CHIP_HEIGHT)),
            "area": box["area"],
            "aspect": (
                (box["x2"] - box["x1"])
                /
                max(1, box["y2"] - box["y1"])
            ),
        })

    # --------------------------------------------------------
    # Match detected rectangles to A/B/C/D using their
    # expected physical dimensions.
    # --------------------------------------------------------

    expected = {
        "A": 40 * 30,
        "B": 50 * 35,
        "C": 30 * 30,
        "D": 40 * 40,
    }

    # Match largest/shape-aware regions
    # B is generally the largest rectangular region,
    # D is the next large square,
    # A is the medium rectangle,
    # C is the smaller square.

    remaining = placements.copy()
    result = {}

    # B: largest area
    b = max(remaining, key=lambda p: p["area"])
    result["B"] = b
    remaining.remove(b)

    # D: largest remaining
    d = max(remaining, key=lambda p: p["area"])
    result["D"] = d
    remaining.remove(d)

    # A: among remaining, more rectangular
    a = max(
        remaining,
        key=lambda p: abs(p["aspect"] - (40 / 30))
    )

    result["A"] = a
    remaining.remove(a)

    # C: final
    result["C"] = remaining[0]

    final = {}

    for name in ["A", "B", "C", "D"]:

        final[name] = {
            "x": round(
                result[name]["x"],
                2
            ),
            "y": round(
                result[name]["y"],
                2
            ),
        }

    return final


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidate(initial):

    candidate = {}

    for name in MACROS:

        # Keep candidate reasonably close to uploaded
        # starting placement.
        dx = random.randint(-25, 25)
        dy = random.randint(-25, 25)

        max_x = CHIP_WIDTH - MACROS[name]["w"]
        max_y = CHIP_HEIGHT - MACROS[name]["h"]

        x = clamp(
            round(initial[name]["x"] + dx),
            0,
            max_x
        )

        y = clamp(
            round(initial[name]["y"] + dy),
            0,
            max_y
        )

        candidate[name] = {
            "x": x,
            "y": y,
        }

    return candidate


# ============================================================
# LOCAL REFINEMENT
# ============================================================

def refine_placement(placement):

    current = {
        name: {
            "x": placement[name]["x"],
            "y": placement[name]["y"],
        }
        for name in MACROS
    }

    current_score = actual_qor(current)

    evaluations = 0

    for _ in range(20):

        improved = False

        for name in MACROS:

            for dx, dy in [
                (5, 0),
                (-5, 0),
                (0, 5),
                (0, -5),
            ]:

                candidate = {
                    n: {
                        "x": current[n]["x"],
                        "y": current[n]["y"],
                    }
                    for n in MACROS
                }

                max_x = CHIP_WIDTH - MACROS[name]["w"]
                max_y = CHIP_HEIGHT - MACROS[name]["h"]

                candidate[name]["x"] = clamp(
                    current[name]["x"] + dx,
                    0,
                    max_x
                )

                candidate[name]["y"] = clamp(
                    current[name]["y"] + dy,
                    0,
                    max_y
                )

                score = actual_qor(candidate)

                evaluations += 1

                if score > current_score:

                    current = candidate
                    current_score = score
                    improved = True

        if not improved:
            break

    return current, current_score, evaluations


# ============================================================
# DRAW FLOORPLAN
# ============================================================

def draw_floorplan(placement):

    WIDTH = 900
    HEIGHT = 900

    image = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        "white"
    )

    draw = ImageDraw.Draw(image)

    left = 90
    top = 40
    right = 850
    bottom = 800

    def px(x):
        return left + (x / CHIP_WIDTH) * (right - left)

    def py(y):
        return bottom - (y / CHIP_HEIGHT) * (bottom - top)

    # Grid
    for value in range(0, 201, 20):

        x = px(value)

        draw.line(
            [(x, top), (x, bottom)],
            fill=(220, 220, 220),
            width=1
        )

        y = py(value)

        draw.line(
            [(left, y), (right, y)],
            fill=(220, 220, 220),
            width=1
        )

    # Chip boundary
    draw.rectangle(
        [left, top, right, bottom],
        outline="black",
        width=3
    )

    colors = {
        "A": (154, 193, 221),
        "B": (154, 193, 221),
        "C": (154, 193, 221),
        "D": (154, 193, 221),
    }

    for name in ["A", "B", "C", "D"]:

        x = placement[name]["x"]
        y = placement[name]["y"]

        w = MACROS[name]["w"]
        h = MACROS[name]["h"]

        x1 = px(x)
        y1 = py(y + h)

        x2 = px(x + w)
        y2 = py(y)

        draw.rectangle(
            [x1, y1, x2, y2],
            fill=colors[name],
            outline="black",
            width=2
        )

        # Label
        try:
            font = ImageFont.truetype(
                "arial.ttf",
                22
            )
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox(
            (0, 0),
            name,
            font=font
        )

        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        draw.text(
            (
                (x1 + x2 - tw) / 2,
                (y1 + y2 - th) / 2,
            ),
            name,
            fill="black",
            font=font
        )

    # Axis labels
    draw.text(
        (450, 820),
        "X Coordinate",
        fill="black"
    )

    draw.text(
        (20, 400),
        "Y Coordinate",
        fill="black"
    )

    return image


def image_to_base64(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return "data:image/png;base64," + encoded


# ============================================================
# OPTIMIZATION
# ============================================================

def optimize_floorplan(initial):

    initial_score = actual_qor(initial)

    candidates = []

    # Include uploaded floorplan itself
    candidates.append(initial)

    # Generate candidate floorplans
    for _ in range(NUM_CANDIDATES - 1):
        candidates.append(
            generate_candidate(initial)
        )

    # Convert to model features
    feature_rows = [
        create_features(c)
        for c in candidates
    ]

    X = pd.DataFrame(feature_rows)

    # ML prediction
    predictions = model.predict(X)

    # Select top predicted candidates
    ranked_indices = np.argsort(
        predictions
    )[::-1][:TOP_K]

    # Actual evaluation
    best = None
    best_score = -float("inf")

    for index in ranked_indices:

        candidate = candidates[index]

        score = actual_qor(candidate)

        if score > best_score:

            best_score = score
            best = candidate

    # Local refinement
    refined, refined_score, refinement_evaluations = (
        refine_placement(best)
    )

    total_evaluations = (
        TOP_K
        +
        refinement_evaluations
    )

    evaluation_reduction = (
        1
        -
        total_evaluations / NUM_CANDIDATES
    ) * 100

    predicted_best = float(
        predictions[ranked_indices[0]]
    )

    optimized_image = draw_floorplan(
        refined
    )

    return {
        "initial_placement": initial,
        "optimized_placement": refined,

        "initial_qor": round(
            initial_score,
            4
        ),

        "qor_before_refinement": round(
            best_score,
            4
        ),

        "qor_after_refinement": round(
            refined_score,
            4
        ),

        "improvement": round(
            refined_score - initial_score,
            4
        ),

        "refinement_improvement": round(
            refined_score - best_score,
            4
        ),

        "predicted_best_qor": round(
            predicted_best,
            4
        ),

        "candidates_generated": NUM_CANDIDATES,

        "ml_screening_evaluations": TOP_K,

        "refinement_evaluations": refinement_evaluations,

        "total_evaluations": total_evaluations,

        "evaluation_reduction": round(
            evaluation_reduction,
            2
        ),

        "optimized_image": image_to_base64(
            optimized_image
        ),
    }


# ============================================================
# API ROUTES
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "service": "AI Floorplanning Optimizer",
    }


@app.get("/api/health")
def health():

    return {
        "status": "healthy",
        "model": "floorplan_model_v3",
        "sklearn_model_loaded": True,
    }


@app.post("/api/optimize")
async def optimize_endpoint(
    file: UploadFile = File(...)
):

    try:

        contents = await file.read()

        if not contents:
            return {
                "success": False,
                "error": "Uploaded file is empty."
            }

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        # Detect initial macro positions
        initial = detect_macros_from_image(
            image
        )

        # Run ML optimization
        result = optimize_floorplan(
            initial
        )

        result["success"] = True

        result["filename"] = file.filename

        return result

    except Exception as e:

        return {
            "success": False,
            "error": str(e),
        }