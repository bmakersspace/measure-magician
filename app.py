from flask import Flask, request, send_file
import cv2
import numpy as np
import io
import math
from sklearn.cluster import AgglomerativeClustering
import zipfile
from flask import Response

app = Flask(__name__, static_url_path='', static_folder='static')

@app.route('/')
def index():
    return app.send_static_file('index.html')

def find_system_bounds(gray_img, expected_system_count=5):
    hist = cv2.reduce(gray_img, 1, cv2.REDUCE_AVG).flatten()
    threshold = 230
    dark_rows = np.where(hist < threshold)[0]

    if len(dark_rows) == 0:
        h = gray_img.shape[0]
        return [(i * h // expected_system_count, (i + 1) * h // expected_system_count)
                for i in range(expected_system_count)]

    dark_rows_reshaped = dark_rows.reshape(-1, 1)
    clustering = AgglomerativeClustering(n_clusters=expected_system_count).fit(dark_rows_reshaped)

    system_bounds = []
    for i in range(expected_system_count):
        cluster_rows = dark_rows[clustering.labels_ == i]
        system_bounds.append((cluster_rows.min(), cluster_rows.max()))

    system_bounds.sort(key=lambda x: x[0])
    return system_bounds

def measure_bar_total_thickness(gray_img, x, top_staff, bottom_staff, darkness_threshold=100, slice_width=20):
    y_start = top_staff
    y_end = bottom_staff
    half_width = slice_width // 2

    x_start = max(x - half_width, 0)
    x_end = min(x + half_width + 1, gray_img.shape[1])

    # Extract vertical slice of the system image
    slice_img = gray_img[y_start:y_end, x_start:x_end]

    # Binarize by darkness threshold (dark = True)
    binary = slice_img < darkness_threshold

    # Sum black pixels column-wise (vertically)
    col_sum = np.sum(binary, axis=0)

    # Consider columns with more than 50% dark pixels as dark columns
    dark_cols = col_sum > ((y_end - y_start) / 2)

    # Find consecutive runs of dark columns and sum their widths
    total_thickness = 0
    current_run = 0
    for val in dark_cols:
        if val:
            current_run += 1
        else:
            if current_run > 0:
                total_thickness += current_run
                current_run = 0
    if current_run > 0:
        total_thickness += current_run

    return total_thickness

def detect_barlines_with_thickness_filter(gray, top_staff, bottom_staff, min_height=100):
    edges = cv2.Canny(gray, 30, 120, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=40,
        maxLineGap=10
    )

    if lines is None:
        return []

    xs = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        height = abs(y2 - y1)
        if abs(angle) > 80 and height >= min_height and x1 > 20:
            xs.append((x1 + x2) // 2)

    xs = sorted(xs)
    print(f"Detected barlines: {xs}")

    # Cluster to remove duplicates
    clustered = []
    min_spacing = 25
    for x in xs:
        if not clustered or abs(x - clustered[-1]) > min_spacing:
            clustered.append(x)

    print(f"Final barlines after filtering: {clustered}")

    # Check thickness for each barline using new total thickness method
    bars_with_thickness = []
    for x in clustered:
        thickness = measure_bar_total_thickness(gray, x, top_staff, bottom_staff)
        is_thick = thickness > 6  # Tune this threshold if needed
        bars_with_thickness.append((x, is_thick))
        print(f"Bar at x={x}: total thickness={thickness} pixels -> {'THICK' if is_thick else 'thin'}")

    # Filter out repeat bars (thick + thin pairs within close proximity)
    final_bars = []
    used_indices = set()
    for i, (x_i, thick_i) in enumerate(bars_with_thickness):
        if i in used_indices:
            continue
        is_repeat = False
        for j in range(i + 1, len(bars_with_thickness)):
            x_j, thick_j = bars_with_thickness[j]
            if abs(x_i - x_j) < 14 and thick_i != thick_j:
                print(f"Excluding repeat pair: x={x_i} and x={x_j}")
                used_indices.add(i)
                used_indices.add(j)
                is_repeat = True
                break
        if not is_repeat:
            final_bars.append((x_i, thick_i))

    print(f"Bars after repeat filtering: {[x for x, _ in final_bars]}")

    return final_bars

def find_staff_vertical_bounds(system_img):
    hist = cv2.reduce(system_img, 1, cv2.REDUCE_AVG).flatten()
    threshold = 200
    staff_rows = np.where(hist < threshold)[0]
    if len(staff_rows) == 0:
        return 0, system_img.shape[0]
    return staff_rows[0], staff_rows[-1]

@app.route("/upload", methods=["POST"])
def process_image():
    uploaded_files = request.files.getlist("image")
    clef_type = request.form.get("clef_type", "double")

    min_bar_height = 100 if clef_type == "double" else 40
    processed_images = []
    
    global_measure_count = request.form.get("measure_start", None)
    measure_count = int(global_measure_count) if global_measure_count else 1  # <-- stay persistent

    for file in uploaded_files:
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        system_bounds = find_system_bounds(gray, expected_system_count=5)

        for top, bottom in system_bounds:
            system_img = gray[top:bottom]
            top_staff, bottom_staff = find_staff_vertical_bounds(system_img)

            barlines = detect_barlines_with_thickness_filter(system_img, top_staff, bottom_staff, min_bar_height)

            if barlines:
                barlines = barlines[:-1]  # Remove rightmost

            prev_x = -100
            min_dist = 15

            for x, is_thick in barlines:
                if is_thick:
                    continue
                if x - prev_x > min_dist:
                    cv2.line(img, (x, top + top_staff), (x, top + bottom_staff), (255, 0, 0), 1)
                    cv2.putText(
                        img,
                        str(measure_count),
                        (x + 5, top + top_staff - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                    measure_count += 1
                    prev_x = x

        _, img_encoded = cv2.imencode('.png', img)
        processed_images.append(img_encoded.tobytes())

    # Return single PNG if only one file was uploaded
    if len(processed_images) == 1:
        return send_file(io.BytesIO(processed_images[0]), mimetype='image/png')

    # Prepare zip archive in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for idx, img_bytes in enumerate(processed_images):
            zip_file.writestr(f"numbered_page_{idx + 1}.png", img_bytes)

    zip_buffer.seek(0)

    return Response(
        zip_buffer,
        mimetype='application/zip',
        headers={
            'Content-Disposition': 'attachment; filename=numbered_pages.zip'
        }
    )
if __name__ == "__main__":
    app.run(debug=True)
