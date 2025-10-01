from flask import Flask, request, send_file, make_response
import cv2
import numpy as np
import io
import os
import requests
import uuid

# Roboflow API
ROBOFLOW_API_KEY = "vYtPQ7BRfeDKCfXexpT5"  # Replace with your API key
MODEL_SINGLE = "mmv3-ati6o/7"  # Replace with your model ID
MODEL_DOUBLE = "mmv3-ati6o/7"  # Replace with your model ID
API_URL = "https://detect.roboflow.com"

app = Flask(__name__, static_url_path='', static_folder='static')

@app.route('/')
def index():
    return app.send_static_file('index.html')

def crop_horizontal_whitespace(gray_img, threshold=250):
    col_means = cv2.reduce(gray_img, 0, cv2.REDUCE_AVG).flatten()
    dark_cols = np.where(col_means < threshold)[0]
    if len(dark_cols) == 0:
        return 0, gray_img.shape[1]
    left, right = dark_cols.min(), dark_cols.max()
    return left-15, right

def find_staff_lines(gray_img, left_crop=0, right_crop=None, darkness_threshold=130, max_line_thickness=4):
    if right_crop is None:
        right_crop = gray_img.shape[1]
    cropped = gray_img[:, left_crop:right_crop]
    row_avgs = np.mean(cropped, axis=1)
    dark_rows = np.where(row_avgs < darkness_threshold)[0]
    lines = []
    start = None
    prev = None
    for r in dark_rows:
        if start is None:
            start = r
            prev = r
        elif r <= prev + 1 and (r - start) < max_line_thickness:
            prev = r
        else:
            lines.append((start, prev))
            start = r
            prev = r
    if start is not None:
        lines.append((start, prev))
    return [(s + e) // 2 for s, e in lines]

def group_staff_lines(staff_lines, lines_per_system=5, max_line_spacing=50, min_lines_per_system=3):
    systems = []
    staff_lines = sorted(staff_lines)
    temp_group = [staff_lines[0]]
    for line in staff_lines[1:]:
        if line - temp_group[-1] <= max_line_spacing:
            temp_group.append(line)
        else:
            if len(temp_group) >= min_lines_per_system:
                systems.append((temp_group[0] - 35, temp_group[-1] + 35))
            temp_group = [line]
    if len(temp_group) >= min_lines_per_system:
        systems.append((temp_group[0] - 35, temp_group[-1] + 35))
    return systems

def detect_barlines_ai_debug(image_bgr, model_id):
    """Send image to Roboflow and draw detection boxes for debugging."""
    height, width = image_bgr.shape[:2]

    # Encode & send to Roboflow
    _, img_encoded = cv2.imencode(".jpg", image_bgr)
    resp = requests.post(
        f"{API_URL}/{model_id}?api_key={ROBOFLOW_API_KEY}&format=json",
        files={"file": img_encoded.tobytes()}
    )
    data = resp.json()
    print("\n--- Roboflow Response ---")
    print(data)  # Full raw JSON for debugging

    # Draw detections on a copy of the image
    debug_img = image_bgr.copy()

    for pred in data.get("predictions", []):
        px = int(pred["x"])
        py = int(pred["y"])
        pw = int(pred["width"])
        ph = int(pred["height"])

        x1 = px - pw // 2
        y1 = py - ph // 2
        x2 = px + pw // 2
        y2 = py + ph // 2

        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(debug_img, pred.get("class", "obj"), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    return data

def extract_barlines_from_response(data, img_width, crop_width):
    """Return list of x-coordinates in the original cropped image scale."""
    x_positions = []
    for pred in data.get("predictions", []):
        # Scale the x position from Roboflow output to our cropped image
        x_scaled = int(pred["x"] * (crop_width / data["image"]["width"]))
        x_positions.append(x_scaled)
    return sorted(x_positions)

@app.route('/upload', methods=['POST'])
def process_image():
    clef_type = request.form.get("clef_type", "single").lower()  # dropdown from frontend

    if clef_type == "single":
        model_id = MODEL_SINGLE
        lines_per_system = 5
    else:
        model_id = MODEL_DOUBLE
        lines_per_system = 10

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Crop horizontal whitespace
    left, right = crop_horizontal_whitespace(gray)
    cropped_gray = gray[:, left:right]

    # Find and group staff lines
    staff_lines = find_staff_lines(cropped_gray, left_crop=0, right_crop=right - left)
    systems = group_staff_lines(staff_lines, lines_per_system=lines_per_system)

    # ✅ Get starting measure number ONCE
    measure_start_raw = request.form.get("measure_start", "1")
    try:
        measure_start = int(measure_start_raw)
    except ValueError:
        measure_start = 1

    measure_count = measure_start - 1  # global counter

    for idx, (top, bottom) in enumerate(systems):
        # Crop system region for AI
        system_crop_bgr = image[top:bottom, left:right]

        # Detect barlines with AI (returns list of x-coordinates)
        data = detect_barlines_ai_debug(system_crop_bgr, model_id)
        bars_x = extract_barlines_from_response(
            data,
            system_crop_bgr.shape[1],
            system_crop_bgr.shape[1]
        )

        # Annotate barlines and measure numbers
        for x in bars_x:
            measure_count += 1
            measure_num = measure_count

            pos = (int(x + left - 20), int(top + 15))  # position label

            # Draw white border
            cv2.putText(
                image,
                str(measure_num),
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                10,
                cv2.LINE_AA
            )

            # Draw black text
            cv2.putText(
                image,
                str(measure_num),
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

    # Encode and return
    _, buffer = cv2.imencode('.png', image)
    io_buf = io.BytesIO(buffer)
    response = make_response(send_file(io_buf, mimetype='image/png', as_attachment=False))
    response.headers['X-Measure-Count'] = str(measure_count + 1)
    return response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Railway provides this automatically
    app.run(host="0.0.0.0", port=port)
    app.run(debug=True)
