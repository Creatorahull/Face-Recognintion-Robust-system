from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from PIL import Image
import io

from predict import predict_image

app = FastAPI()


# -----------------------------
# BASE TEMPLATE (Reusable UI)
# -----------------------------
def base_template(content):
    return f"""
    <html>
    <head>
        <title>Face Recognition</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #0f172a;
                color: white;
                margin: 0;
            }}

            .navbar {{
                display: flex;
                justify-content: center;
                gap: 30px;
                padding: 15px;
                background: #020617;
            }}

            .navbar a {{
                color: #38bdf8;
                text-decoration: none;
                font-weight: bold;
            }}

            .navbar a:hover {{
                color: #0ea5e9;
            }}

            .container {{
                display: flex;
                justify-content: center;
                align-items: center;
                height: calc(100vh - 60px);
            }}

            .card {{
                background: #1e293b;
                padding: 30px;
                border-radius: 12px;
                width: 400px;
                text-align: center;
                box-shadow: 0 0 20px rgba(0,0,0,0.4);
            }}

            .upload-box {{
                border: 2px dashed #38bdf8;
                padding: 20px;
                border-radius: 10px;
                cursor: pointer;
                margin-bottom: 15px;
            }}

            .upload-box:hover {{
                background: #0f172a;
            }}

            img {{
                max-width: 100%;
                margin-top: 10px;
                border-radius: 10px;
            }}

            button {{
                margin-top: 15px;
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                background: #38bdf8;
                color: black;
                font-weight: bold;
                cursor: pointer;
            }}

            button:hover {{
                background: #0ea5e9;
            }}

            .result {{
                margin-top: 15px;
                font-size: 16px;
            }}

            .loader {{
                display: none;
                margin-top: 10px;
            }}
        </style>
    </head>

    <body>

        <div class="navbar">
            <a href="/">Home</a>
            <a href="/about">About</a>
            <a href="/contact">Contact</a>
        </div>

        <div class="container">
            <div class="card">
                {content}
            </div>
        </div>

        <script>
            let selectedFile = null;

            function previewImage(event) {{
                selectedFile = event.target.files[0];
                const preview = document.getElementById("preview");
                if (preview) {{
                    preview.src = URL.createObjectURL(selectedFile);
                }}
            }}

            async function sendImage() {{
                if (!selectedFile) {{
                    alert("Please select an image first.");
                    return;
                }}

                const loader = document.getElementById("loader");
                const resultDiv = document.getElementById("result");

                loader.style.display = "block";
                resultDiv.innerText = "";

                let formData = new FormData();
                formData.append("file", selectedFile);

                try {{
                    const res = await fetch("/predict", {{
                        method: "POST",
                        body: formData
                    }});

                    const data = await res.json();

                    loader.style.display = "none";

                    resultDiv.innerText =
                        "Class: " + data.class_name +
                        " | Confidence: " + data.confidence.toFixed(3);

                }} catch (err) {{
                    loader.style.display = "none";
                    resultDiv.innerText = "Error: " + err;
                }}
            }}
        </script>

    </body>
    </html>
    """


# -----------------------------
# HOME PAGE
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return base_template("""
        <h2>Face Recognition</h2>

        <div class="upload-box" onclick="document.getElementById('fileInput').click()">
            Drag & Drop or Click to Upload
            <input type="file" id="fileInput" hidden onchange="previewImage(event)">
        </div>

        <img id="preview" />

        <button onclick="sendImage()">Predict</button>

        <div class="loader" id="loader">Processing...</div>
        <div class="result" id="result"></div>
    """)


# -----------------------------
# ABOUT PAGE
# -----------------------------
@app.get("/about", response_class=HTMLResponse)
def about():
    return base_template("""
        <h2>About</h2>

        <p><b>Creator:</b> Rahull</p>

        <p>
        I am an AI Engineer working on deep learning and computer vision projects.
        This project uses a custom-trained face recognition model to identify individuals.
        </p>

        <p>
        It also detects unknown people using a confidence threshold,
        making it more robust in real-world scenarios.
        </p>
    """)


# -----------------------------
# CONTACT PAGE
# -----------------------------
@app.get("/contact", response_class=HTMLResponse)
def contact():
    return base_template("""
        <h2>Contact</h2>
        <p>Email: <b>creatorahull@gmail.com</b></p>
    """)


# -----------------------------
# PREDICTION ENDPOINT
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    result = predict_image(image)

    return {
        "class_name": result["class"],
        "confidence": result["confidence"]
    }