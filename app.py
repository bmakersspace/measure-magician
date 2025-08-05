from flask import Flask, request, send_file
import cv2
import numpy as np
import io
import math

app = Flask(__name__, static_url_path='', static_folder='static')

@app.route('/')
def index():
    return app.send_static_file('index.html')

def detect_barlines(gray):
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=100,
        minLineLength=int(gray.shape[0] * 0.6),
        maxLineGap=10
    )
    xs = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:,0]:
            angle = math.degrees(math.atan2(y2-y1, x2-x1))
            if abs(angle) > 80:  # near-vertical
                xs.append((x1 + x2)//2)
    # cluster nearby x’s
    xs = sorted(xs)
    clustered = []
    for x in xs:
        if not clustered or abs(x - clustered[-1]) > 20:
            clustered.append(x)
    return clustered

@app.route("/upload", methods=["POST"])
def process_image():
    file = request.files['image']
    np_img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    xs = detect_barlines(gray)
    xs = [0] + xs

    hist = cv2.reduce(gray, 1, cv2.REDUCE_AVG).flatten()
    staff_top = next((i for i, v in enumerate(hist) if v < 250), 30)

    for i, x in enumerate(xs):
        cv2.putText(
            img,
            str(i+1),
            (x + 5, max(staff_top - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    _, img_encoded = cv2.imencode('.png', img)
    return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/png')
    
if __name__ == "__main__":
    app.run(debug=True)
