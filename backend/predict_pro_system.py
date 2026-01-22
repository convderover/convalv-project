import os
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import yaml
import torch.nn as nn
from pathlib import Path
from PIL import Image
from scipy.signal import hilbert, find_peaks
from torchvision import transforms

# --- 1. ARQUITECTURA DE LA RED (Sincronizada con Entrenamiento V4) ---
class ConvalvHolisticNet(nn.Module):
    def __init__(self, num_acoustic_metrics=3, num_clinical_metrics=2):
        super(ConvalvHolisticNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.num_mlp = nn.Sequential(
            nn.Linear(num_acoustic_metrics + num_clinical_metrics, 64),
            nn.ReLU(), 
            nn.Linear(64, 64), 
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 + 64, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, image, numeric_data):
        v_feat = self.cnn(image)
        n_feat = self.num_mlp(numeric_data)
        return self.classifier(torch.cat((v_feat, n_feat), dim=1))

# --- 2. CONFIGURACIÓN DE RUTAS Y PARÁMETROS ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "cleaned_audio"
AI_READY_DIR = BASE_DIR / "data" / "ai_ready"
MODEL_PATH = BASE_DIR / "models" / "best_convalv_holistic_v4.pth"

SR_NATIVO = 2000
DURACION_OBJETIVO = 20.0
IA_RES = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for d in [CLEAN_DIR, AI_READY_DIR]: d.mkdir(parents=True, exist_ok=True)

# --- 3. MOTOR DE LIMPIEZA QUIRÚRGICA PRO ---
def procesar_cirugia_pro(y, sr):
    # A. Filtrado Espectral (20-400Hz)
    stft = librosa.stft(y, n_fft=2048, hop_length=64)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    stft[(freqs < 20) | (freqs > 400), :] = 0
    y_filt = librosa.istft(stft, hop_length=64)

    # B. Detección de Envolvente y Picos
    env = np.abs(hilbert(y_filt))
    env = np.convolve(env, np.ones(50)/50, mode='same')
    pks, props = find_peaks(env, distance=int(0.2*sr), prominence=np.mean(env)*0.6)
    
    # C. Limpieza de Outliers de Amplitud
    if len(pks) > 0:
        mediana_p = np.median(props['prominences'])
        for i, p in enumerate(pks):
            if props['prominences'][i] > mediana_p * 2.5:
                start = max(0, p - int(0.1*sr))
                end = min(len(y_filt), p + int(0.1*sr))
                y_filt[start:end] = 0

    # D. Gate Rítmico (Silencio entre latidos) - CON FALLBACK
    if len(pks) >= 2:
        mask = np.zeros_like(y_filt)
        for p in pks:
            start = max(0, p - int(0.15*sr))  # 150ms antes
            end = min(len(mask), p + int(0.15*sr))  # 150ms después
            mask[start:end] = 1.0
        y_clean = y_filt * mask
    else:
        # Si no hay suficientes picos, usar el audio filtrado sin gate
        y_clean = y_filt
        print("⚠️ Pocos picos detectados, omitiendo gate rítmico")

    y_clean = librosa.util.normalize(y_clean) * 0.98
    
    # E. Lógica de Etiquetado S1/S2 (Basada en Intervalos RR)
    labels = []
    for i in range(len(pks)-1):
        diff = (pks[i+1] - pks[i]) / sr
        if i == 0:
            labels.append("S1" if diff < 0.44 else "S2")
        else:
            labels.append("S2" if labels[-1] == "S1" else "S1")
    if len(pks) > 0:
        labels.append("S2" if not labels or labels[-1] == "S1" else "S1")

    
    # F. Cálculo de Métricas CORREGIDO (Basado en pares S1-S2)
        # Agrupar picos en pares S1-S2 (un latido = S1 + S2)
        latidos_s1 = []
        i = 0
        while i < len(pks) - 1:
            t1 = pks[i] / sr
            t2 = pks[i+1] / sr
            dt = t2 - t1
            
            # Si el intervalo está entre 200-500ms, es un par S1-S2 válido
            if 0.20 < dt < 0.50:
                latidos_s1.append(pks[i])  # Guardar solo el S1 de cada latido
                i += 2  # Saltar al siguiente par
            else:
                i += 1  # Pico ruidoso, buscar siguiente

        # Calcular R-R entre S1 consecutivos (latido a latido)
        if len(latidos_s1) > 1:
            intervalos_rr = np.diff(latidos_s1) / sr
            bpm = float(60 / np.mean(intervalos_rr))
            rvv = float(np.std(intervalos_rr) * 1000)
        else:
            bpm = 75.0
            rvv = 50.0

        print(f"🫀 Latidos válidos detectados: {len(latidos_s1)} | BPM corregido: {bpm:.1f}")
        
        return y_clean, env, pks, labels, bpm, rvv

# --- 4. FUNCIÓN PRINCIPAL DE DIAGNÓSTICO ---
def ejecutar_diagnostico():
    archivos = [f for f in os.listdir(RAW_DIR) if f.endswith(('.wav', '.m4a', '.mp3'))]
    if not archivos:
        return print("❌ RAW_DIR vacío. Sube archivos a data/raw")
    
    for i, f in enumerate(archivos): print(f" [{i}] {f}")
    idx = int(input(f"👉 Selecciona el audio para analizar: "))
    
    nombre_base = archivos[idx].split('.')[0]
    ruta_audio = str(RAW_DIR / archivos[idx])
    
    # Carga de Audio
    y_raw, sr = librosa.load(ruta_audio, sr=SR_NATIVO, duration=DURACION_OBJETIVO)
    if len(y_raw) < DURACION_OBJETIVO * SR_NATIVO:
        y_raw = np.pad(y_raw, (0, int(DURACION_OBJETIVO * SR_NATIVO) - len(y_raw)))
    
    # Procesamiento PRO
    y_clean, env, pks, lbs, bpm, rvv = procesar_cirugia_pro(y_raw, SR_NATIVO)
    
    # Guardar Audio Limpio
    sf.write(CLEAN_DIR / f"pro_clean_{nombre_base}.wav", y_clean, SR_NATIVO)
    
    # Generar Imagen Maestra para la IA
    plt.figure(figsize=(IA_RES/100, IA_RES/100), dpi=100)
    S_ia = librosa.feature.melspectrogram(y=y_clean, sr=SR_NATIVO, n_mels=IA_RES, fmax=500)
    librosa.display.specshow(librosa.power_to_db(S_ia, ref=np.max), fmax=500, cmap='magma')
    plt.axis('off')
    path_img = AI_READY_DIR / f"{nombre_base}_master.png"
    plt.savefig(path_img, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Guardar Reporte YAML (Limpio de binarios NumPy)
    data_yaml = {
        "biometria": {
            "bpm": float(bpm), 
            "rr_variability": float(rvv), 
            "snr_clean": float(np.mean(y_clean**2))
        },
        "clinica": {
            "edad": 50, 
            "genero": "Male"
        }
    }
    with open(AI_READY_DIR / f"report_{nombre_base}.yaml", 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    # Carga de Modelo y Predicción
    model = ConvalvHolisticNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Preparamos Tensores
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(Image.open(path_img).convert('RGB')).unsqueeze(0).to(DEVICE)
    
    # Escalado idéntico al entrenamiento
    nums = torch.tensor([[bpm/200.0, rvv/500.0, data_yaml['biometria']['snr_clean']*10.0, 0.5, 0.5]], dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        logits = model(img_tensor, nums)
        prob = torch.sigmoid(logits).item()

    # --- VISUALIZACIÓN FINAL ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    plt.style.use('dark_background')
    times = np.linspace(0, DURACION_OBJETIVO, len(y_clean))
    
    # Gráfico de Onda
    ax1.plot(times, y_clean, color='cyan', alpha=0.7)
    for p, l in zip(pks, lbs):
        t = p/SR_NATIVO
        color = 'lime' if l == "S1" else 'orange'
        ax1.text(t, 1.05, l, color=color, ha='center', fontweight='bold')
        ax1.axvline(t, color=color, alpha=0.3, linestyle='--')
    ax1.set_title(f"Análisis CONVALV PRO: {nombre_base} | Ritmo: {bpm:.1f} BPM", color="white")

    # Espectrograma de Diagnóstico
    S_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y_clean, sr=SR_NATIVO, n_mels=128), ref=np.max)
    librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=SR_NATIVO, ax=ax2, cmap='magma')
    
    # Veredicto por Consola
    print("\n" + "="*45)
    color_txt = "\033[92m" if prob > 0.5 else "\033[91m"
    estado = "NORMAL" if prob > 0.5 else "ABNORMAL"
    confianza = prob if prob > 0.5 else (1 - prob)
    print(f"VEREDICTO IA: {color_txt}{estado}\033[0m")
    print(f"CONFIANZA:   {confianza*100:.2f}%")
    print("="*45)
    
    plt.tight_layout()
    plt.show()

def ejecutar_diagnostico_api(audio_path):
    """
    Versión adaptada para la API.
    Retorna: dict con resultados del diagnóstico.
    NO usa input() ni print() como salida principal.
    NO hace plt.show() para evitar bloquear.
    """

    # --- Nombre base ---
    nombre_base = Path(audio_path).stem

    # --- Carga de Audio ---
    y_raw, sr = librosa.load(audio_path, sr=SR_NATIVO, duration=DURACION_OBJETIVO)
    if len(y_raw) < DURACION_OBJETIVO * SR_NATIVO:
        y_raw = np.pad(y_raw, (0, int(DURACION_OBJETIVO * SR_NATIVO) - len(y_raw)))

    # --- Procesamiento PRO ---
    y_clean, env, pks, labels, bpm, rvv = procesar_cirugia_pro(y_raw, SR_NATIVO)

    # --- Guardar Audio Limpio ---
    sf.write(CLEAN_DIR / f"pro_clean_{nombre_base}.wav", y_clean, SR_NATIVO)

    # --- Generar Imagen Maestra para la IA ---
    plt.figure(figsize=(IA_RES/100, IA_RES/100), dpi=100)
    S_ia = librosa.feature.melspectrogram(y=y_clean, sr=SR_NATIVO, n_mels=IA_RES, fmax=500)
    librosa.display.specshow(librosa.power_to_db(S_ia, ref=np.max), fmax=500, cmap='magma')
    plt.axis('off')
    path_img = AI_READY_DIR / f"{nombre_base}_master.png"
    plt.savefig(path_img, bbox_inches='tight', pad_inches=0)
    plt.close()

    # --- Guardar Reporte YAML (Limpio de binarios NumPy) ---
    data_yaml = {
        "biometria": {
            "bpm": float(bpm),
            "rr_variability": float(rvv),
            "snr_clean": float(np.mean(y_clean**2))
        },
        "clinica": {
            "edad": 50,
            "genero": "Male"
        }
    }
    with open(AI_READY_DIR / f"report_{nombre_base}.yaml", 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    # --- Carga de Modelo y Predicción ---
    model = ConvalvHolisticNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # --- Preparamos Tensores ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(Image.open(path_img).convert('RGB')).unsqueeze(0).to(DEVICE)

    # --- Escalado idéntico al entrenamiento ---
    nums = torch.tensor(
        [[bpm/200.0, rvv/500.0, data_yaml['biometria']['snr_clean']*10.0, 0.5, 0.5]],
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor, nums)
        prob = torch.sigmoid(logits).item()

    diagnosis = "NORMAL" if prob > 0.5 else "ABNORMAL"
    confianza = prob if prob > 0.5 else (1 - prob)

    # ✅ Retorno limpio para API / Frontend
    return {
        "audio_path": str(audio_path),
        "nombre_base": nombre_base,
        "clean_audio_path": str(CLEAN_DIR / f"pro_clean_{nombre_base}.wav"),
        "master_image_path": str(path_img),
        "yaml_report_path": str(AI_READY_DIR / f"report_{nombre_base}.yaml"),
        "bpm": float(bpm),
        "rvv": float(rvv),
        "prob": float(prob),
        "confianza": float(confianza),
        "diagnosis": diagnosis,
        "pks": pks.tolist() if hasattr(pks, "tolist") else list(pks),
        "labels": labels
    }

if __name__ == "__main__":
    ejecutar_diagnostico()
