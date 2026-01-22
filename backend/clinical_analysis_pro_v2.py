import os
import yaml
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.signal import hilbert, find_peaks, welch, savgol_filter, butter, sosfilt
from scipy.stats import entropy, skew, kurtosis
from datetime import datetime
import warnings

# --- CONFIGURACIÓN DE INTERFAZ DE ALTA FIDELIDAD ---
warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.style.use('dark_background')

# Paleta de colores: Estándar Monitor de Hospital
UI = {
    'bg': '#050505',
    'pcg': '#00E5FF',      # Cian Quirúrgico
    'ecg': '#FF003C',      # Rojo Pulso
    's1': '#39FF14',       # Verde Bio (Válvulas AV)
    's2': '#FFCC00',       # Oro Valvular (Válvulas Semilunares)
    'envelope': '#BF5AF2', # Púrpura Energía
    'grid': '#1C1C1E',
    'text': '#F2F2F7',
    'accent': '#5E5CE6'    # Azul Indigo
}

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "cleaned_audio"
AI_READY_DIR = BASE_DIR / "data" / "ai_ready"
REPORTS_DIR = BASE_DIR / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# =================================================================
# SECCIÓN 1: MOTOR DE BIOINGENIERÍA (LÓGICA MÉDICA)
# =================================================================

class CardiacEngine:
    """Motor de procesamiento avanzado para validación de ciclos cardíacos."""

    @staticmethod
    def extract_validated_beats(peaks_t):
        """
        Lógica de Emparejamiento Quirúrgico:
        Define 1 Latido = Par S1-S2.
        Calcula R-R = S1_actual - S1_anterior.
        """
        s1_validated, s2_validated = [], []
        systoles, rr_intervals = [], []
        
        i = 0
        while i < len(peaks_t) - 1:
            t1 = peaks_t[i]
            t2 = peaks_t[i+1]
            dt = t2 - t1
            
            # FILTRO FISIOLÓGICO: Sístole (S1->S2) debe estar entre 200ms y 500ms
            if 0.20 < dt < 0.50:
                s1_validated.append(t1)
                s2_validated.append(t2)
                systoles.append(dt * 1000) # En ms
                
                # Cálculo de R-R (S1 a S1)
                if len(s1_validated) > 1:
                    rr_val = (s1_validated[-1] - s1_validated[-2]) * 1000
                    rr_intervals.append(rr_val)
                i += 2 # Salto al siguiente par funcional
            else:
                i += 1 # Pico ruidoso detectado, buscar siguiente candidato
        
        return s1_validated, s2_validated, systoles, rr_intervals

    @staticmethod
    def filter_outliers_iqr(data):
        """Elimina latidos espurios usando el Rango Intercuartílico."""
        if len(data) < 5: return data
        data_arr = np.array(data)
        q1 = np.percentile(data_arr, 25)
        q3 = np.percentile(data_arr, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return data_arr[(data_arr > lower) & (data_arr < upper)].tolist()

    @staticmethod
    def generate_ecg_simulation(s1_times, s2_times, duration, fs):
        """Genera ondas P-QRS-T sincrónicas con cada latido validado."""
        t = np.linspace(0, duration, int(duration * fs))
        ecg = np.zeros_like(t)
        
        for ts1 in s1_times:
            # Complejo QRS (S1 ocurre al inicio de la sístole)
            qrs_p = ts1 - 0.02
            ecg += 1.5 * np.exp(-((t - qrs_p)/0.01)**2)    # R
            ecg -= 0.3 * np.exp(-((t - (qrs_p-0.015))/0.008)**2) # Q
            ecg -= 0.4 * np.exp(-((t - (qrs_p+0.025))/0.012)**2) # S
            # Onda P
            p_p = qrs_p - 0.15
            ecg += 0.2 * np.exp(-((t - p_p)/0.025)**2)
            
        for ts2 in s2_times:
            # Onda T (Ocurre tras la eyección, cerca de S2)
            t_p = ts2 + 0.04
            ecg += 0.4 * np.exp(-((t - t_p)/0.05)**2)
            
        return ecg

# =================================================================
# SECCIÓN 2: LABORATORIO ANALÍTICO (CLINICAL SUITE)
# =================================================================

class ClinicalLabV6:
    def __init__(self, filename):
        self.filename = filename
        self.base_id = filename.replace('pro_clean_', '').replace('.wav', '')
        self.y, self.sr = librosa.load(CLEAN_DIR / filename, sr=2000)
        self.dur = len(self.y) / self.sr
        
        with open(AI_READY_DIR / f"report_{self.base_id}.yaml", 'r') as f:
            self.meta = yaml.safe_load(f)

    def run_analysis_pipeline(self):
        """Ejecuta la cadena de procesamiento de biometría."""
        # 1. Envolvente de Energía Hilbert Suavizada
        self.env = np.abs(hilbert(self.y))
        self.env = savgol_filter(self.env, 71, 3)
        
        # 2. Peak Picking base con periodo refractario de 250ms
        pks, _ = find_peaks(self.env, distance=int(0.25*self.sr), 
                           prominence=np.mean(self.env)*0.7)
        peaks_t = pks / self.sr
        
        # 3. Validación Quirúrgica de Ciclos
        v_s1, v_s2, v_sys, v_rr = CardiacEngine.extract_validated_beats(peaks_t)
        
        # 4. Limpieza de Outliers (R-R y Sístole)
        self.s1_t = v_s1
        self.s2_t = v_s2
        self.sys_ms = CardiacEngine.filter_outliers_iqr(v_sys)
        self.rr_ms = CardiacEngine.filter_outliers_iqr(v_rr)
        
        # 5. Métricas Espectrales y ECG
        self.ecg_sim = CardiacEngine.generate_ecg_simulation(self.s1_t, self.s2_t, self.dur, self.sr)
        self.psd_f, self.psd_p = welch(self.y, self.sr, nperseg=1024)
        self.zcr = librosa.feature.zero_crossing_rate(self.y)[0]
        self.bpm = 60000 / np.mean(self.rr_ms) if self.rr_ms else 0

    # --- MONITOR I: TEMPORAL Y ELECTRO-MECÁNICA ---
    def open_temporal_monitor(self):
        fig = plt.figure(figsize=(16, 10), facecolor=UI['bg'])
        fig.canvas.manager.set_window_title(f'MONITOR I: TEMPORAL - {self.base_id}')
        gs = gridspec.GridSpec(3, 1, height_ratios=[1.5, 1, 0.8], hspace=0.45)
        t = np.linspace(0, self.dur, len(self.y))

        # PCG Waveform
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(t, self.y, color=UI['pcg'], alpha=0.15, label='Audio PCG Limpio')
        ax1.plot(t, self.env, color=UI['envelope'], linewidth=2, label='Envolvente Bio-Energía')
        ax1.set_title("ANÁLISIS DE ENVOLVENTE Y GOLPE VALVULAR", color=UI['envelope'], loc='left', fontsize=14, fontweight='bold')
        ax1.grid(color=UI['grid'], alpha=0.4)
        ax1.legend(loc='upper right', frameon=False)

        # ECG Sincronizado
        

#[Image of the electrical conduction system of the heart]

        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(t, self.ecg_sim, color=UI['ecg'], linewidth=1.5, label='Simulación PQRST (1 QRS = 1 Latido)')
        ax2.set_title("SINCRONÍA ELECTRO-VALVULAR (S1-QRS)", color=UI['ecg'], loc='left')
        ax2.grid(color=UI['grid'], alpha=0.4)
        ax2.legend(loc='upper right', frameon=False)

        # Mapeo de S1-S2
        
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        for s1 in self.s1_t: ax3.axvline(s1, color=UI['s1'], linewidth=2.5, label='S1 (V. Mitral)' if s1==self.s1_t[0] else "")
        for s2 in self.s2_t: ax3.axvline(s2, color=UI['s2'], linewidth=2.5, label='S2 (V. Aórtica)' if s2==self.s2_t[0] else "", linestyle=':')
        ax3.set_title("DETECCIÓN DE EVENTOS VALVULARES (Sin Solapamientos)", color=UI['text'], loc='left')
        ax3.set_xlabel("Tiempo (s)")
        ax3.legend(loc='upper right', frameon=False)

        plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.08)
        plt.show(block=False)

    # --- MONITOR II: ANÁLISIS ESPECTRAL Y TURBULENCIA ---
    def open_spectral_monitor(self):
        fig = plt.figure(figsize=(16, 10), facecolor=UI['bg'])
        fig.canvas.manager.set_window_title(f'MONITOR II: ESPECTRAL - {self.base_id}')
        gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.25)

        # Spectrogram
        ax1 = fig.add_subplot(gs[0, :])
        
        S = librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_mels=256, fmax=500, n_fft=1024)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', 
                                 sr=self.sr, fmax=500, ax=ax1, cmap='magma')
        ax1.set_title("ESPECTROGRAMA MEL DE ALTA DENSIDAD", color='white', fontsize=14, fontweight='bold')

        # Power Spectral Density
        
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.semilogy(self.psd_f, self.psd_p, color=UI['s1'], linewidth=1.5)
        ax2.fill_between(self.psd_f, self.psd_p, color=UI['s1'], alpha=0.2)
        ax2.set_xlim(20, 500)
        ax2.set_title("DENSIDAD DE POTENCIA (Análisis de Soplos)", color=UI['text'])
        ax2.grid(color=UI['grid'])

        # Zero Crossing Rate (Fricción)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(self.zcr, color=UI['pcg'], linewidth=1)
        ax3.set_title("TASA DE CRUCE POR CERO (Complejidad de Fricción)", color=UI['text'])
        ax3.grid(color=UI['grid'])

        plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.08)
        plt.show(block=False)

    # --- MONITOR III: DINÁMICA RÍTMICA (CLEANED) ---
    def open_rhythm_monitor(self):
        fig = plt.figure(figsize=(16, 10), facecolor=UI['bg'])
        fig.canvas.manager.set_window_title(f'MONITOR III: RÍTMICA - {self.base_id}')
        gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

        # Poincaré Plot (Estabilidad RR)
        
        ax1 = fig.add_subplot(gs[0, 0])
        if len(self.rr_ms) > 2:
            rr_n = np.array(self.rr_ms[:-1])
            rr_n1 = np.array(self.rr_ms[1:])
            ax1.scatter(rr_n, rr_n1, color=UI['pcg'], edgecolors='white', s=80, alpha=0.6)
            ax1.plot([min(rr_n), max(rr_n)], [min(rr_n), max(rr_n)], 'r--', alpha=0.3)
        ax1.set_title("MAPA DE POINCARÉ (R-R Dinámico Sin Outliers)", color=UI['text'])
        ax1.set_xlabel("RR_n (ms)")
        ax1.set_ylabel("RR_n+1 (ms)")

        # Histograma de Sístoles
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(self.sys_ms, bins=15, color=UI['s2'], alpha=0.6, edgecolor='white')
        ax2.set_title("DISTRIBUCIÓN DE DURACIÓN SISTÓLICA (S1-S2)", color=UI['text'])

        # Boxplot Comparativo
        ax3 = fig.add_subplot(gs[1, :])
        ax3.boxplot([self.rr_ms, self.sys_ms], vert=False, patch_artist=True,
                    boxprops=dict(facecolor=UI['accent'], alpha=0.5))
        ax3.set_yticklabels(['Intervalos R-R', 'Intervalos S1-S2'])
        ax3.set_title("ANÁLISIS DE DISPERSIÓN TEMPORAL", color=UI['text'])
        ax3.grid(color=UI['grid'], axis='x')

        plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1)
        plt.show(block=False)

    # --- MONITOR IV: REPORTE BIO-CLÍNICO ---
    def open_final_report(self):
        fig = plt.figure(figsize=(10, 12), facecolor='#0A0A0A')
        fig.canvas.manager.set_window_title('MONITOR IV: INFORME FINAL')
        ax = fig.add_subplot(111); ax.axis('off')

        bio = self.meta.get('biometria', {})
        cli = self.meta.get('clinica', {})

        final_txt = (
            f"      SUITE ANALÍTICA CARDIOVASCULAR CONVALV V6.0\n"
            f"      {'='*52}\n"
            f"      ID PACIENTE:  {self.base_id.upper()}\n"
            f"      FECHA ESTUDIO: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"      {'='*52}\n\n"
            f"      [1. BIOMETRÍA RÍTMICA (Filtrada)]\n"
            f"      Ritmo Cardíaco:        {self.bpm:.1f} BPM\n"
            f"      Intervalo R-R Medio:   {np.mean(self.rr_ms):.1f} ms\n"
            f"      Variabilidad (SDNN):   {np.std(self.rr_ms):.2f} ms\n\n"
            f"      [2. DINÁMICA DE VÁLVULAS]\n"
            f"      Sístole (S1-S2) Media: {np.mean(self.sys_ms):.1f} ms\n"
            f"      Diástole Proyectada:   {(np.mean(self.rr_ms) - np.mean(self.sys_ms)):.1f} ms\n"
            f"      Relación Síst/Diást:   {(np.mean(self.sys_ms)/(np.mean(self.rr_ms)-np.mean(self.sys_ms))):.3f}\n\n"
            f"      [3. CARACTERIZACIÓN ESPECTRAL]\n"
            f"      Entropía de Shannon:   {entropy(self.y):.4f}\n"
            f"      Frecuencia Dominante:  {self.psd_f[np.argmax(self.psd_p)]:.1f} Hz\n"
            f"      Asimetría (Skewness):  {skew(self.y):.4f}\n\n"
            f"      [4. PERFIL CLÍNICO REGISTRADO]\n"
            f"      Edad del Paciente:     {cli.get('edad', 'N/A')} años\n"
            f"      Género Registrado:     {cli.get('genero', 'N/A')}\n"
            f"      SNR de Señal:          {bio.get('snr_clean', 0):.6f}\n\n"
            f"      {'='*52}\n"
            f"      DIAGNÓSTICO BASADO EN BIOMECÁNICA CARDÍACA"
        )

        ax.text(0.1, 0.95, final_txt, family='monospace', fontsize=12, color=UI['s1'], 
                verticalalignment='top', bbox=dict(facecolor='#121212', edgecolor=UI['grid'], boxstyle='round,pad=2'))

        plt.savefig(REPORTS_DIR / f"V6_REPORT_{self.base_id}.png", dpi=200, bbox_inches='tight')
        plt.show()

# =================================================================
# LANZADOR PRINCIPAL
# =================================================================

def main():
    print("\n" + "█"*60)
    print("      CONVALV FORENSIC CLINICAL LABORATORY V6.0")
    print("█"*60)
    
    # Listar archivos procesados
    files = [f for f in os.listdir(CLEAN_DIR) if f.startswith('pro_clean_')]
    if not files:
        print("❌ Error: No se detectaron archivos procesados en data/cleaned_audio.")
        return

    for i, f in enumerate(files): print(f" [{i}] {f}")
    
    try:
        idx = int(input("\n👉 Selecciona el ID de paciente para el estudio V6: "))
        lab = ClinicalLabV6(files[idx])
        
        print(f"\n🧠 Analizando '{files[idx]}' con motor de emparejamiento valvular...")
        lab.run_analysis_pipeline()
        
        print("📡 Desplegando Estaciones de Diagnóstico (Monitor I a IV)...")
        lab.open_temporal_monitor()
        lab.open_spectral_monitor()
        lab.open_rhythm_monitor()
        lab.open_final_report()
        
    except Exception as e:
        print(f"❌ Error durante el análisis clínico: {e}")

# =====================================================================================
# ============================ API EXTENSIONS (NO TOCAR) ===============================
# =====================================================================================

# ✅ AÑADIDO PARA API: guardar figuras en memoria como PNG bytes
import io

def open_temporal_monitor_api(self):
    """Versión que retorna la imagen como bytes en lugar de mostrarla."""
    fig = plt.figure(figsize=(16, 10), facecolor=UI['bg'])
    fig.canvas.manager.set_window_title(f'MONITOR I: TEMPORAL - {self.base_id}')
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.5, 1, 0.8], hspace=0.45)
    t = np.linspace(0, self.dur, len(self.y))

    # PCG Waveform
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, self.y, color=UI['pcg'], alpha=0.15, label='Audio PCG Limpio')
    ax1.plot(t, self.env, color=UI['envelope'], linewidth=2, label='Envolvente Bio-Energía')
    ax1.set_title("ANÁLISIS DE ENVOLVENTE Y GOLPE VALVULAR", color=UI['envelope'], loc='left', fontsize=14, fontweight='bold')
    ax1.grid(color=UI['grid'], alpha=0.4)
    ax1.legend(loc='upper right', frameon=False)

    # ECG Sincronizado
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t, self.ecg_sim, color=UI['ecg'], linewidth=1.5, label='Simulación PQRST (1 QRS = 1 Latido)')
    ax2.set_title("SINCRONÍA ELECTRO-VALVULAR (S1-QRS)", color=UI['ecg'], loc='left')
    ax2.grid(color=UI['grid'], alpha=0.4)
    ax2.legend(loc='upper right', frameon=False)

    # Mapeo de S1-S2
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    for s1 in self.s1_t: ax3.axvline(s1, color=UI['s1'], linewidth=2.5, label='S1 (V. Mitral)' if s1==self.s1_t[0] else "")
    for s2 in self.s2_t: ax3.axvline(s2, color=UI['s2'], linewidth=2.5, label='S2 (V. Aórtica)' if s2==self.s2_t[0] else "", linestyle=':')
    ax3.set_title("DETECCIÓN DE EVENTOS VALVULARES (Sin Solapamientos)", color=UI['text'], loc='left')
    ax3.set_xlabel("Tiempo (s)")
    ax3.legend(loc='upper right', frameon=False)

    plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.08)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, facecolor=UI['bg'])
    buf.seek(0)
    plt.close()
    return buf.getvalue()

def open_spectral_monitor_api(self):
    """Versión que retorna la imagen como bytes en lugar de mostrarla."""
    fig = plt.figure(figsize=(16, 10), facecolor=UI['bg'])
    fig.canvas.manager.set_window_title(f'MONITOR II: ESPECTRAL - {self.base_id}')
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.25)

    # Spectrogram
    ax1 = fig.add_subplot(gs[0, :])
    
    S = librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_mels=256, fmax=500, n_fft=1024)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', 
                             sr=self.sr, fmax=500, ax=ax1, cmap='magma')
    ax1.set_title("ESPECTROGRAMA MEL DE ALTA DENSIDAD", color='white', fontsize=14, fontweight='bold')

    # Power Spectral Density
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.semilogy(self.psd_f, self.psd_p, color=UI['s1'], linewidth=1.5)
    ax2.fill_between(self.psd_f, self.psd_p, color=UI['s1'], alpha=0.2)
    ax2.set_xlim(20, 500)
    ax2.set_title("DENSIDAD DE POTENCIA (Análisis de Soplos)", color=UI['text'])
    ax2.grid(color=UI['grid'])

    # Zero Crossing Rate (Fricción)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(self.zcr, color=UI['pcg'], linewidth=1)
    ax3.set_title("TASA DE CRUCE POR CERO (Complejidad de Fricción)", color=UI['text'])
    ax3.grid(color=UI['grid'])

    plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.08)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, facecolor=UI['bg'])
    buf.seek(0)
    plt.close()
    return buf.getvalue()

def open_rhythm_monitor_api(self):
    """Versión que retorna la imagen como bytes en lugar de mostrarla."""
    fig = plt.figure(figsize=(16, 10), facecolor=UI['bg'])
    fig.canvas.manager.set_window_title(f'MONITOR III: RÍTMICA - {self.base_id}')
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Poincaré Plot (Estabilidad RR)
    ax1 = fig.add_subplot(gs[0, 0])
    if len(self.rr_ms) > 2:
        rr_n = np.array(self.rr_ms[:-1])
        rr_n1 = np.array(self.rr_ms[1:])
        ax1.scatter(rr_n, rr_n1, color=UI['pcg'], edgecolors='white', s=80, alpha=0.6)
        ax1.plot([min(rr_n), max(rr_n)], [min(rr_n), max(rr_n)], 'r--', alpha=0.3)
    ax1.set_title("MAPA DE POINCARÉ (R-R Dinámico Sin Outliers)", color=UI['text'])
    ax1.set_xlabel("RR_n (ms)")
    ax1.set_ylabel("RR_n+1 (ms)")

    # Histograma de Sístoles
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(self.sys_ms, bins=15, color=UI['s2'], alpha=0.6, edgecolor='white')
    ax2.set_title("DISTRIBUCIÓN DE DURACIÓN SISTÓLICA (S1-S2)", color=UI['text'])

    # Boxplot Comparativo
    ax3 = fig.add_subplot(gs[1, :])
    ax3.boxplot([self.rr_ms, self.sys_ms], vert=False, patch_artist=True,
                boxprops=dict(facecolor=UI['accent'], alpha=0.5))
    ax3.set_yticklabels(['Intervalos R-R', 'Intervalos S1-S2'])
    ax3.set_title("ANÁLISIS DE DISPERSIÓN TEMPORAL", color=UI['text'])
    ax3.grid(color=UI['grid'], axis='x')

    plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, facecolor=UI['bg'])
    buf.seek(0)
    plt.close()
    return buf.getvalue()

def open_final_report_api(self):
    """Versión que retorna la imagen como bytes en lugar de mostrarla."""
    fig = plt.figure(figsize=(10, 12), facecolor='#0A0A0A')
    fig.canvas.manager.set_window_title('MONITOR IV: INFORME FINAL')
    ax = fig.add_subplot(111); ax.axis('off')

    bio = self.meta.get('biometria', {})
    cli = self.meta.get('clinica', {})

    final_txt = (
        f"      SUITE ANALÍTICA CARDIOVASCULAR CONVALV V6.0\n"
        f"      {'='*52}\n"
        f"      ID PACIENTE:  {self.base_id.upper()}\n"
        f"      FECHA ESTUDIO: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"      {'='*52}\n\n"
        f"      [1. BIOMETRÍA RÍTMICA (Filtrada)]\n"
        f"      Ritmo Cardíaco:        {self.bpm:.1f} BPM\n"
        f"      Intervalo R-R Medio:   {np.mean(self.rr_ms):.1f} ms\n"
        f"      Variabilidad (SDNN):   {np.std(self.rr_ms):.2f} ms\n\n"
        f"      [2. DINÁMICA DE VÁLVULAS]\n"
        f"      Sístole (S1-S2) Media: {np.mean(self.sys_ms):.1f} ms\n"
        f"      Diástole Proyectada:   {(np.mean(self.rr_ms) - np.mean(self.sys_ms)):.1f} ms\n"
        f"      Relación Síst/Diást:   {(np.mean(self.sys_ms)/(np.mean(self.rr_ms)-np.mean(self.sys_ms))):.3f}\n\n"
        f"      [3. CARACTERIZACIÓN ESPECTRAL]\n"
        f"      Entropía de Shannon:   {entropy(self.y):.4f}\n"
        f"      Frecuencia Dominante:  {self.psd_f[np.argmax(self.psd_p)]:.1f} Hz\n"
        f"      Asimetría (Skewness):  {skew(self.y):.4f}\n\n"
        f"      [4. PERFIL CLÍNICO REGISTRADO]\n"
        f"      Edad del Paciente:     {cli.get('edad', 'N/A')} años\n"
        f"      Género Registrado:     {cli.get('genero', 'N/A')}\n"
        f"      SNR de Señal:          {bio.get('snr_clean', 0):.6f}\n\n"
        f"      {'='*52}\n"
        f"      DIAGNÓSTICO BASADO EN BIOMECÁNICA CARDÍACA"
    )

    ax.text(0.1, 0.95, final_txt, family='monospace', fontsize=12, color=UI['s1'], 
            verticalalignment='top', bbox=dict(facecolor='#121212', edgecolor=UI['grid'], boxstyle='round,pad=2'))

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='#0A0A0A')
    buf.seek(0)
    plt.close()
    return buf.getvalue()

def run_full_report_api(filename):
    """
    Ejecuta todo el pipeline clínico y retorna:
    - métricas importantes
    - imágenes PNG en bytes para monitor I-IV
    """
    lab = ClinicalLabV6(filename)
    lab.run_analysis_pipeline()

    return {
        "base_id": lab.base_id,
        "bpm": float(lab.bpm),
        "rr_mean_ms": float(np.mean(lab.rr_ms)) if lab.rr_ms else 0.0,
        "rr_std_ms": float(np.std(lab.rr_ms)) if lab.rr_ms else 0.0,
        "systole_mean_ms": float(np.mean(lab.sys_ms)) if lab.sys_ms else 0.0,
        "entropy_shannon": float(entropy(lab.y)),
        "monitor_1_png": lab.open_temporal_monitor_api(),
        "monitor_2_png": lab.open_spectral_monitor_api(),
        "monitor_3_png": lab.open_rhythm_monitor_api(),
        "monitor_4_png": lab.open_final_report_api()
    }

# ✅ Monkey patch: añadir métodos API a la clase SIN EDITAR su bloque original
ClinicalLabV6.open_temporal_monitor_api = open_temporal_monitor_api
ClinicalLabV6.open_spectral_monitor_api = open_spectral_monitor_api
ClinicalLabV6.open_rhythm_monitor_api = open_rhythm_monitor_api
ClinicalLabV6.open_final_report_api = open_final_report_api

if __name__ == "__main__":
    main()
