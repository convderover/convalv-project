# backend_api.py - API Flask para conectar tus scripts con la interfaz React
import soundfile as sf
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import base64
import io
from pathlib import Path
import torch
import librosa
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI
import matplotlib.pyplot as plt
from scipy.signal import welch

# Importar tus funciones de los scripts
# Asegúrate de tener predict_pro_system.py y clinical_analysis_pro_v2.py accesibles
import predict_pro_system as predictor
import clinical_analysis_pro_v2 as clinical

# AÑADIR ESTAS LÍNEAS:
# Ruta del modelo
MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_convalv_holistic_v4.pth"


app = Flask(__name__)
CORS(app)  # Permitir peticiones desde React

@app.route("/", methods=["GET"])
def home():
    return "CONVALV backend OK ✅", 200


# Configuración
UPLOAD_FOLDER = Path('./uploads')
TEMP_FOLDER = Path('./temp')
UPLOAD_FOLDER.mkdir(exist_ok=True)
TEMP_FOLDER.mkdir(exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# ============================================================
# ENDPOINT 1: PREDICCIÓN (predict_pro_system.py)
# ============================================================
@app.route('/api/predict', methods=['POST'])
def predict_audio():
    """
    Recibe un archivo de audio y ejecuta el diagnóstico completo.
    """
    try:
        print("=" * 50)
        print("📥 Recibiendo petición de diagnóstico...")
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No se recibió archivo de audio'}), 400
        
        audio_file = request.files['audio']
        print(f"✅ Archivo recibido: {audio_file.filename}")
        
        # Guardar archivo temporalmente
        temp_path = UPLOAD_FOLDER / f"temp_{audio_file.filename}"
        audio_file.save(temp_path)
        print(f"💾 Guardado en: {temp_path}")
        
        # ============ PROCESAMIENTO REAL ============
        print("🔬 Iniciando procesamiento de audio...")
        
        # 1. Cargar y procesar audio
        y_raw, sr = librosa.load(temp_path, sr=2000, duration=20.0)
        if len(y_raw) < 20 * 2000:
            y_raw = np.pad(y_raw, (0, int(20 * 2000) - len(y_raw)))
        print("✅ Audio cargado")
        
        # 2. Procesamiento PRO (tu función)
        y_clean, env, pks, labels, bpm, rvv = predictor.procesar_cirugia_pro(y_raw, 2000)
        print(f"✅ Audio procesado - BPM: {bpm:.1f}")

        # 3. Guardar audio limpio como archivo estático
        import unicodedata
        import re

        # Sanitizar nombre de archivo (quitar acentos y caracteres especiales)
        original_name = audio_file.filename.replace(' ', '_')
        # Normalizar unicode y quitar acentos
        normalized = unicodedata.normalize('NFKD', original_name)
        ascii_name = normalized.encode('ASCII', 'ignore').decode('ASCII')
        # Quitar caracteres no alfanuméricos excepto guiones bajos, puntos y guiones
        safe_name = re.sub(r'[^\w.\-]', '', ascii_name)
        clean_audio_filename = f"cleaned_{safe_name}.wav"

        clean_audio_path = TEMP_FOLDER / clean_audio_filename

        # Normalizar el audio antes de guardar (evitar audio silencioso)
        # Normalizar el audio antes de guardar (evitar audio silencioso)
        # Normalizar el audio
        y_normalized = y_clean / np.max(np.abs(y_clean)) * 0.9
        y_normalized = np.clip(y_normalized, -1.0, 1.0)

        # Resamplear a 44100 Hz para compatibilidad con navegadores
        y_resampled = librosa.resample(y_normalized, orig_sr=2000, target_sr=44100)

        print(f"🔊 DEBUG Audio - Max abs: {np.max(np.abs(y_normalized)):.6f}")
        print(f"🔊 DEBUG Audio - Mean abs: {np.mean(np.abs(y_normalized)):.6f}")
        print(f"🔊 DEBUG Audio - Non-zero samples: {np.count_nonzero(y_normalized)} / {len(y_normalized)}")
        print(f"🔊 DEBUG Audio - Picos detectados: {len(pks)}")
        print(f"🔊 DEBUG Audio - Resampleado a 44100 Hz: {len(y_resampled)} muestras")

        # Guardar con sample rate estándar
        sf.write(clean_audio_path, y_resampled.astype(np.float32), 44100, subtype='PCM_16')

        # URL directa al archivo
        clean_audio_url = request.host_url.rstrip("/") + f"/temp/{clean_audio_filename}"

        print(f"✅ Audio limpio guardado: {clean_audio_path}")
        print(f"✅ URL: {clean_audio_url}")
        # NO BORRAR el archivo todavía, lo necesitamos disponible
        # clean_audio_path.unlink()  ← COMENTA ESTA LÍNEA
        
        # 4. Generar imagen para IA
        S_ia = librosa.feature.melspectrogram(y=y_clean, sr=2000, n_mels=224, fmax=500)
        
        plt.figure(figsize=(2.24, 2.24), dpi=100)
        librosa.display.specshow(librosa.power_to_db(S_ia), fmax=500, cmap='magma')
        plt.axis('off')
        img_path = TEMP_FOLDER / "temp_master.png"
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print("✅ Espectrograma generado")
        
        # 5. Cargar modelo y predecir
        model = predictor.ConvalvHolisticNet().to(predictor.DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=predictor.DEVICE))        
        model.eval()
        print("✅ Modelo cargado")
        
        # Preparar tensores
        from PIL import Image as PILImage
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(PILImage.open(img_path).convert('RGB')).unsqueeze(0).to(predictor.DEVICE)
        snr = float(np.mean(y_clean**2))
        nums = torch.tensor([[bpm/200.0, rvv/500.0, snr*10.0, 0.5, 0.5]], dtype=torch.float32).to(predictor.DEVICE)
        
        with torch.no_grad():
            logits = model(img_tensor, nums)
            prob = torch.sigmoid(logits).item()
        
        diagnosis = "NORMAL" if prob > 0.5 else "ABNORMAL"
        confidence = (prob * 100) if prob > 0.5 else ((1 - prob) * 100)
        print(f"✅ Diagnóstico IA: {diagnosis} ({confidence:.1f}%)")
        
        # 6. Generar gráficos visuales
        print("📊 Generando gráficos...")
        graphs = generate_diagnostic_graphs(y_clean, env, pks, labels, bpm, rvv, 2000)
        print("✅ Gráficos generados")
        
        # 7. Calcular métricas adicionales
        intervalos = np.diff(pks) / 2000 * 1000  # ms
        rr_mean = float(np.mean(intervalos)) if len(intervalos) > 0 else 0
        rr_std = float(np.std(intervalos)) if len(intervalos) > 0 else 0
        
        # Calcular sístole/diástole
        systoles = []
        for i in range(len(pks)-1):
            if i < len(labels) and labels[i] == "S1" and i+1 < len(labels) and labels[i+1] == "S2":
                systoles.append((pks[i+1] - pks[i]) / 2000 * 1000)
        
        systole_mean = float(np.mean(systoles)) if systoles else 300.0
        diastole_mean = rr_mean - systole_mean if rr_mean > 0 else 500.0
        
        # Entropía espectral
        from scipy.stats import entropy as scipy_entropy
        signal_entropy = float(scipy_entropy(np.abs(y_clean)))
        
        # Usar welch en lugar de spectral_contrast
        from scipy.signal import welch
        psd_f, psd_p = welch(y_clean, 2000, nperseg=1024)
        dominant_freq = float(psd_f[np.argmax(psd_p)]) if len(psd_f) > 0 else 0
        
        # 8. Preparar respuesta JSON
        response = {
            'diagnosis': diagnosis,
            'confidence': round(confidence, 1),
            'biometrics': {
                'bpm': round(float(bpm), 1),
                'rr_mean': round(rr_mean, 1),
                'rr_variability': round(rr_std, 2),
                'systole_mean': round(systole_mean, 1),
                'diastole_mean': round(diastole_mean, 1)
            },
            'spectral': {
                'entropy': round(signal_entropy, 4),
                'dominant_freq': round(dominant_freq, 1)
            },
            'events': {
                's1_count': labels.count("S1"),
                's2_count': labels.count("S2")
            },
            'graphs': graphs,
            'cleaned_audio_url': clean_audio_url
        }
        
        # Limpiar archivos temporales
        temp_path.unlink()
        
        print("✅ Diagnóstico completado")
        print("=" * 50)
        return jsonify(response), 200
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ ERROR CRÍTICO: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 50)
        return jsonify({'error': str(e)}), 500


# ============================================================
# FUNCIÓN AUXILIAR: GENERAR GRÁFICOS
# ============================================================
def generate_diagnostic_graphs(y_clean, env, pks, labels, bpm, rvv, sr):
    """
    Genera los 4 gráficos de monitores y los retorna como base64.
    """
    graphs = {}
    
    # --- MONITOR I: TEMPORAL ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor('#050505')
    times = np.linspace(0, 20, len(y_clean))
    
    # Onda + Envolvente
    ax1.plot(times, y_clean, color='#00E5FF', alpha=0.3)
    ax1.plot(times, env, color='#BF5AF2', linewidth=2)
    ax1.set_facecolor('#050505')
    ax1.grid(color='#1C1C1E', alpha=0.3)
    ax1.set_title('Envolvente Bio-Energía', color='white')
    
    # ECG simulado
    ecg_sim = np.zeros_like(y_clean)
    for p in pks:
        if p < len(ecg_sim):
            ecg_sim[max(0, p-10):min(len(ecg_sim), p+10)] += np.random.randn(min(20, len(ecg_sim)-max(0,p-10)))
    ax2.plot(times, ecg_sim, color='#FF003C', linewidth=1.5)
    ax2.set_facecolor('#050505')
    ax2.set_title('Simulación ECG', color='white')
    
    # Marcadores S1/S2
    for p, l in zip(pks, labels):
        t = p / sr
        color = '#39FF14' if l == "S1" else '#FFCC00'
        ax3.axvline(t, color=color, linewidth=2, alpha=0.6)
    ax3.set_facecolor('#050505')
    ax3.set_xlabel('Tiempo (s)', color='white')
    ax3.set_title('Eventos Valvulares', color='white')
    
    plt.tight_layout()
    graphs['temporal_url'] = fig_to_base64(fig)
    plt.close()
    
    # --- MONITOR II: ESPECTRAL ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.patch.set_facecolor('#050505')
    
    S = librosa.feature.melspectrogram(y=y_clean, sr=sr, n_mels=256, fmax=500)
    librosa.display.specshow(librosa.power_to_db(S), x_axis='time', y_axis='mel', 
                             sr=sr, fmax=500, ax=ax1, cmap='magma')
    ax1.set_title('Espectrograma MEL', color='white')
    
    from scipy.signal import welch
    psd_f, psd_p = welch(y_clean, sr, nperseg=1024)
    ax2.semilogy(psd_f.flatten(), psd_p.flatten(), color='#39FF14')
    ax2.set_facecolor('#050505')
    ax2.set_title('Densidad de Potencia', color='white')
    ax2.grid(color='#1C1C1E')
    
    plt.tight_layout()
    graphs['spectral_url'] = fig_to_base64(fig)
    plt.close()
    
    # --- MONITOR III: RÍTMICA ---
    intervalos = np.diff(pks) / sr * 1000
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('#050505')
    
    if len(intervalos) > 2:
        ax1.scatter(intervalos[:-1], intervalos[1:], color='#00E5FF', s=80, alpha=0.6)
        ax1.plot([min(intervalos), max(intervalos)], [min(intervalos), max(intervalos)], 'r--', alpha=0.3)
    ax1.set_facecolor('#050505')
    ax1.set_title('Mapa Poincaré', color='white')
    ax1.set_xlabel('RR_n (ms)', color='white')
    ax1.set_ylabel('RR_n+1 (ms)', color='white')
    ax1.grid(color='#1C1C1E')
    
    ax2.hist(intervalos, bins=15, color='#FFCC00', alpha=0.6, edgecolor='white')
    ax2.set_facecolor('#050505')
    ax2.set_title('Distribución R-R', color='white')
    ax2.grid(color='#1C1C1E')
    
    plt.tight_layout()
    graphs['rhythm_url'] = fig_to_base64(fig)
    plt.close()
    
    # --- MONITOR IV: REPORTE (puedes añadir un resumen textual) ---
    # Este gráfico puede ser generado por clinical_analysis_pro_v2.py
    graphs['report_url'] = None  # Opcional
    
    return graphs


def fig_to_base64(fig):
    """Convierte una figura de matplotlib a base64 para enviar al frontend."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


@app.route("/", methods=["GET"])
def home():
    return "CONVALV backend OK", 200


# ============================================================
# ENDPOINT 2: ANÁLISIS CLÍNICO (clinical_analysis_pro_v2.py)
# ============================================================
@app.route('/api/clinical-analysis', methods=['POST'])
def clinical_analysis():
    """
    Ejecuta el análisis clínico detallado y retorna las 4 gráficas.
    """
    try:
        # Aquí integrarías el código de clinical_analysis_pro_v2.py
        # Similar al endpoint anterior pero llamando a las funciones de ese script
        
        # Ejemplo básico:
        # lab = clinical.ClinicalLabV6(filename)
        # lab.run_analysis_pipeline()
        # graphs = {
        #     'temporal': lab.open_temporal_monitor(),
        #     'spectral': lab.open_spectral_monitor(),
        #     ...
        # }
        
        return jsonify({'message': 'Análisis clínico completado'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# ============================================================
# ENDPOINT: SERVIR ARCHIVOS DE AUDIO TEMPORAL
# ============================================================
@app.route('/temp/<filename>')
def serve_temp_file(filename):
    """Sirve archivos temporales de audio"""
    file_path = TEMP_FOLDER / filename
    if file_path.exists():
        return send_file(file_path, mimetype='audio/wav')
    return jsonify({'error': 'Archivo no encontrado'}), 404


# ============================================================
# INICIAR SERVIDOR
# ============================================================
if __name__ == '__main__':
    print("🚀 Servidor CONVALV Backend iniciado en http://localhost:5000")
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)