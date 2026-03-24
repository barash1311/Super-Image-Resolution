import os
import io
import base64
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
from inference import SRInference

app = Flask(__name__)
CORS(app)

# Configuration
CONFIG_PATH = 'config.yaml'
# Search for the latest model
MODEL_DIR = 'experiments/hybrid_sr_v1'
MODEL_PATH = os.path.join(MODEL_DIR, 'latest.pth')

# Initialize Inference
sr_engine = None
try:
    if os.path.exists(MODEL_PATH):
        sr_engine = SRInference(config_path=CONFIG_PATH, model_path=MODEL_PATH)
    else:
        print(f"Warning: Model weights not found at {MODEL_PATH}. App will start but results might be poor.")
        sr_engine = SRInference(config_path=CONFIG_PATH)
except Exception as e:
    print(f"Error initializing SR Engine: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enhance', methods=['POST'])
def enhance():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        # Load Image
        input_image = Image.open(file.stream).convert('RGB')
        
        # Enhance
        output_image = sr_engine.enhance(input_image)
        
        # Convert to Base64 for easy frontend display
        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            'enhanced_image': img_str,
            'message': 'Success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
