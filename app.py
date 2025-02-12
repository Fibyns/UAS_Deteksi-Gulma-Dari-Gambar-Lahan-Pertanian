import os
import re
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

# Path ke folder citra
dataset_path = 'C:/Users/MSyaripudin/Documents/CITRA/'  # Ubah ke path lokal Anda
image_folder = os.path.join(dataset_path, "image")
label_folder = os.path.join(dataset_path, "PixelLabelData")
output_folder = os.path.join(dataset_path, "output_visualization")
os.makedirs(output_folder, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    # Ambil file dari permintaan
    file = request.files['image']
    filename = file.filename
    image_path = os.path.join(image_folder, filename)
    file.save(image_path)

    # Ambil angka terakhir dari nama file untuk mencocokkan label
    base_filename = os.path.splitext(filename)[0]  # Misalnya, 'image10'
    match = re.search(r'\d+$', base_filename)  # Ambil semua angka di akhir string
    label_number = match.group() if match else "0"  # Gunakan angka yang ditemukan
    label_path = os.path.join(label_folder, f"Label_{label_number}.png")

    if not os.path.exists(label_path):
        return jsonify({'error': 'Label file not found'}), 404

    # Baca gambar asli
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Baca label
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    # Buat colormap untuk label
    label_colored = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    colors = {
        0: (0, 0, 0),    # Background
        1: (0, 255, 0),  # Class 1
        2: (255, 0, 0),  # Class 2 (gulma)
        3: (0, 0, 255)   # Class 3
    }

    for class_id, color in colors.items():
        label_colored[label == class_id] = color

    # Hitung jumlah piksel gulma (kelas 2)
    num_weeds = np.sum(label == 2)

    # Gabungkan gambar asli dan label berwarna
    overlay = cv2.addWeighted(image, 0.7, label_colored, 0.3, 0)

    # Simpan hasil visualisasi
    output_path = os.path.join(output_folder, f"result_{filename}")
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Kembalikan URL gambar hasil
    output_file_url = f"/output_visualization/{os.path.basename(output_path)}"
    
    return jsonify({
        'message': f'Processing done! Detected {num_weeds} weeds.',
        'output_file': output_file_url
    })

@app.route('/output_visualization/<path:filename>')
def send_output(filename):
    return send_from_directory(output_folder, filename)

if __name__ == '__main__':
    app.run(debug=True)