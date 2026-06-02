import os
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)
MODEL_PATH = 'best_model_fixed2.keras'

model = None
model_error = ""

try:
    print(f"Keras: {tf.keras.__version__} | TF: {tf.__version__}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model berhasil dimuat!")
except Exception as e:
    model_error = str(e)
    print(f"❌ Gagal: {e}")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'model_error': model_error if model is None else None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'status': 'error', 'message': model_error}), 500
    try:
        data = request.json

        input_dict = {
            'usia':                   tf.constant([[float(data.get('usia', 20.0))]], dtype=tf.float32),
            'durasi_tidur_menit':     tf.constant([[float(data.get('durasi_tidur_menit', 0))]], dtype=tf.float32),
            'screen_sebelum_tidur':   tf.constant([[float(data.get('screen_sebelum_tidur', 0))]], dtype=tf.float32),
            'waktu_outdoor':          tf.constant([[float(data.get('waktu_outdoor', 0))]], dtype=tf.float32),
            'konsentrasi':            tf.constant([[int(data.get('konsentrasi', 1))]], dtype=tf.int64),
            'interaksi_sosial':       tf.constant([[int(data.get('interaksi_sosial', 1))]], dtype=tf.int64),
            'jenis_kelamin':          tf.constant([[str(data.get('jenis_kelamin', 'Laki-laki'))]], dtype=tf.string),
            'pekerjaan':              tf.constant([[str(data.get('pekerjaan', 'Pelajar'))]], dtype=tf.string),
            'sering_terbangun_malam': tf.constant([[str(data.get('sering_terbangun_malam', 'Tidak'))]], dtype=tf.string),
            'mimpi_buruk':            tf.constant([[str(data.get('mimpi_buruk', 'Tidak'))]], dtype=tf.string),
            'minum_kopi_hari_ini':    tf.constant([[str(data.get('minum_kopi_hari_ini', 'Tidak'))]], dtype=tf.string),
            'merokok':                tf.constant([[str(data.get('merokok', 'Tidak'))]], dtype=tf.string),
            'konsumsi_alkohol':       tf.constant([[str(data.get('konsumsi_alkohol', 'Tidak'))]], dtype=tf.string),
            'deadline_hari_ini':      tf.constant([[str(data.get('deadline_hari_ini', 'Tidak'))]], dtype=tf.string),
            'lembur':                 tf.constant([[str(data.get('lembur', 'Tidak'))]], dtype=tf.string),
            'aktivitas_hobi':         tf.constant([[str(data.get('aktivitas_hobi', 'Tidak'))]], dtype=tf.string),
            'suasana_hati':           tf.constant([[str(data.get('suasana_hati', 'Netral'))]], dtype=tf.string),
            'konflik_interpersonal':  tf.constant([[str(data.get('konflik_interpersonal', 'Tidak'))]], dtype=tf.string),
            'merasa_kesepian':        tf.constant([[str(data.get('merasa_kesepian', 'Tidak'))]], dtype=tf.string),
            'meditasi':               tf.constant([[str(data.get('meditasi', 'Tidak'))]], dtype=tf.string),
        }

        prediction = model.predict(input_dict)
        stress_level = int(tf.argmax(prediction[0]).numpy())
        return jsonify({'status': 'success', 'stress_level_result': stress_level})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)