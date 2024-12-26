from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import jittor
from models.modeling import VisionTransformer, CONFIGS

app = Flask(__name__)

# 配置上传目录和生成图片目录
UPLOAD_FOLDER = './uploads'
IMAGE_FOLDER = './images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 配置静态文件夹
app.add_url_rule('/images/<filename>', 'images', build_only=True)
from werkzeug.middleware.shared_data import SharedDataMiddleware
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/images': IMAGE_FOLDER
})

MODEL_CONFIG = "ViT-B_16"
IMG_SIZE = 448
PRETRAINED_MODEL_PATH = "output/cub-non-overlap_checkpoint.bin"

def load_model():
    config = CONFIGS[MODEL_CONFIG]
    config.split = 'non-overlap'
    num_classes = 200
    model = VisionTransformer(config, IMG_SIZE, zero_head=True, num_classes=num_classes)
    model.load_state_dict(jittor.load(PRETRAINED_MODEL_PATH)['model'])
    return model

model = load_model()
model.eval()

def get_class_names(idx):
    class_names = []
    with open('minist/cub.txt', 'r') as f:
        for line in f:
            class_names.append(line.strip())
    return class_names[idx]

@app.route('/')
def index():
    return render_template('index.html', prediction=None, image_path=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected')

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image = preprocess_image(filepath)
    prediction = model(image, save_selected_path=f"images/{filename}")
    prediction_label = np.argmax(prediction.numpy(), axis=-1)
    prediction_class = get_class_names(prediction_label[0])

    return render_template('index.html', prediction=prediction_class, image_path=f"/images/{filename}")


def preprocess_image(filepath):
    from PIL import Image
    image = Image.open(filepath).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image).astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)  # 转为 CHW 格式
    image = np.expand_dims(image, axis=0)  # 添加批次维度
    return jittor.array(image)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
