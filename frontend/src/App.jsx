import React, { useState, useRef, useEffect } from 'react';
import { Heart, Mic, Upload, Play, Pause, RefreshCw, X, ChevronRight, FileAudio, CheckCircle, AlertCircle, TrendingUp, Activity } from 'lucide-react';

// CONFIGURACIÓN: Reemplaza estas URLs con las rutas de tus recursos
const CONFIG = {
  API_URL: 'https://convalv-backend.onrender.com/api',
  LOGO_MAIN: '/assets/convalv_logo.png',
  LOGO_PERSONAL: '/assets/roberto_logo.png',
};

const ConvalvApp = () => {
  const [screen, setScreen] = useState('loading');
  const [audioFile, setAudioFile] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [currentMonitor, setCurrentMonitor] = useState(0);
  const [graphs, setGraphs] = useState({
    temporal: null,
    spectral: null,
    rhythm: null,
    report: null
  });
  const [cleanedAudioUrl, setCleanedAudioUrl] = useState(null);
  const [isPlayingCleaned, setIsPlayingCleaned] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);
  
  const audioRef = useRef(null);
  const cleanedAudioRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordingTimerRef = useRef(null);

  // Función para llamar al backend de Python
  const callPythonAPI = async (endpoint, formData) => {
    try {
      const response = await fetch(`${CONFIG.API_URL}${endpoint}`, {
        method: 'POST',
        body: formData,
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percentCompleted);
        }
      });

      if (!response.ok) {
        throw new Error(`Error del servidor: ${response.status}`);
      }

      return await response.json();
    } catch (err) {
      setError(err.message);
      console.error('Error en la API:', err);
      return null;
    }
  };

  // Simulación de carga de dependencias
  useEffect(() => {
    if (screen === 'loading') {
      const timer = setTimeout(() => setScreen('welcome'), 2000);
      return () => clearTimeout(timer);
    }
  }, [screen]);

  // Verificación del audio limpio
  useEffect(() => {
    if (cleanedAudioUrl) {
        console.log("✅ cleanedAudioUrl:", cleanedAudioUrl);

        if (cleanedAudioRef.current) {
        cleanedAudioRef.current.src = cleanedAudioUrl;
        cleanedAudioRef.current.load();
        }
    }
  }, [cleanedAudioUrl]);


  // Timer de grabación
  useEffect(() => {
    if (isRecording) {
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime(prev => {
          if (prev >= 20) {
            stopRecording();
            return 20;
          }
          return prev + 1;
        });
      }, 1000);
    } else {
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
      setRecordingTime(0);
    }
    return () => {
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
    };
  }, [isRecording]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      const chunks = [];
      
      mediaRecorderRef.current.ondataavailable = (e) => chunks.push(e.data);
      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/wav' });
        setAudioFile(blob);
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (err) {
      alert('Error al acceder al micrófono');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const duration = 25; // Simulación
      if (duration < 20 || duration > 30) {
        alert('El audio debe tener entre 20 y 30 segundos');
        return;
      }
      setAudioFile(file);
      setScreen('preview');
    }
  };

  const togglePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const runDiagnostic = async () => {
    setScreen('processing');
    setError(null);
    
    // Crear FormData con el archivo de audio
    const formData = new FormData();
    formData.append('audio', audioFile);
    
    try {
      // Llamada al script predict_pro_system.py
      const diagnosticResult = await callPythonAPI('/predict', formData);
      
      if (!diagnosticResult) {
        throw new Error('No se recibieron resultados del diagnóstico');
      }
      
      // Guardar resultados
      setAnalysisResults({
        ia_diagnosis: diagnosticResult.diagnosis,
        ia_confidence: diagnosticResult.confidence,
        bpm: diagnosticResult.biometrics.bpm,
        rr_mean: diagnosticResult.biometrics.rr_mean,
        rr_std: diagnosticResult.biometrics.rr_variability,
        systole_mean: diagnosticResult.biometrics.systole_mean,
        diastole_mean: diagnosticResult.biometrics.diastole_mean,
        entropy: diagnosticResult.spectral.entropy,
        dominant_freq: diagnosticResult.spectral.dominant_freq,
        s1_count: diagnosticResult.events.s1_count,
        s2_count: diagnosticResult.events.s2_count
      });
      
      // Guardar las imágenes de los gráficos
      setGraphs({
        temporal: diagnosticResult.graphs.temporal_url,
        spectral: diagnosticResult.graphs.spectral_url,
        rhythm: diagnosticResult.graphs.rhythm_url,
        report: diagnosticResult.graphs.report_url
      });
      
      // Guardar el audio limpio
      // Guardar el audio limpio (limpiar parámetros de query si existen)
      let audioUrl = diagnosticResult.cleaned_audio_url;
      if (audioUrl && audioUrl.includes("?")) {
        audioUrl = audioUrl.split("?")[0];
      }
      setCleanedAudioUrl(audioUrl);
      
      // Avanzar a la pantalla de monitores
      setScreen('monitors');
      
    } catch (err) {
      setError(`Error en el diagnóstico: ${err.message}`);
      setScreen('preview');
    }
  };

  const togglePlayPauseCleaned = () => {
    if (cleanedAudioRef.current) {
      if (isPlayingCleaned) {
        cleanedAudioRef.current.pause();
      } else {
        cleanedAudioRef.current.play();
      }
      setIsPlayingCleaned(!isPlayingCleaned);
    }
  };

  const getFinalDiagnosis = () => {
    if (!analysisResults) return { status: 'INDETERMINADO', color: 'gray', recommendations: [] };
    
    const { bpm, rr_std, systole_mean, entropy, ia_diagnosis, ia_confidence } = analysisResults;
    let issues = [];
    
    // Detectar anomalías (solo para recomendaciones adicionales)
    if (bpm < 50) issues.push('Bradicardia severa detectada');
    if (bpm > 120) issues.push('Taquicardia severa detectada');
    if (rr_std > 150) issues.push('Muy alta variabilidad R-R');
    if (systole_mean < 200 || systole_mean > 450) issues.push('Duración sistólica fuera de rango');
    if (entropy > 2.0) issues.push('Muy alta complejidad espectral');
    
    // EL DIAGNÓSTICO FINAL SIEMPRE SIGUE A LA IA
    if (ia_diagnosis === 'ABNORMAL') {
        return {
        status: 'ANORMAL',
        color: 'red',
        recommendations: [
            `Modelo de IA detectó anomalía (confianza: ${ia_confidence.toFixed(1)}%)`,
            'Consulta con cardiólogo URGENTE recomendada',
            'Realizar electrocardiograma completo',
            'Evaluación de válvulas cardíacas',
            ...issues
        ]
        };
    }
    
    // Si la IA dice NORMAL pero hay muchas anomalías severas
    if (issues.length >= 2) {
        return {
        status: 'REQUIERE ATENCIÓN',
        color: 'yellow',
        recommendations: [
            `Modelo de IA: ${ia_diagnosis} (confianza: ${ia_confidence.toFixed(1)}%)`,
            'Algunos parámetros fuera de rango normal',
            'Monitoreo continuo recomendado',
            'Repetir análisis en 24-48h',
            ...issues
        ]
        };
    }
    
    // Todo normal
    return {
        status: 'NORMAL',
        color: 'green',
        recommendations: [
        `Modelo de IA: ${ia_diagnosis} (confianza: ${ia_confidence.toFixed(1)}%)`,
        'Parámetros dentro de rangos aceptables',
        'Mantener chequeos periódicos',
        'Estilo de vida saludable',
        ...(issues.length > 0 ? ['Nota: ' + issues.join(', ')] : [])
        ]
      };
  };

  // PANTALLA DE CARGA
  if (screen === 'loading') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <Heart className="w-24 h-24 text-red-500 mx-auto mb-6 animate-pulse" />
          <div className="text-white text-2xl font-bold mb-4">CONVALV</div>
          <div className="text-cyan-300 text-sm mb-6">Inicializando módulos de diagnóstico...</div>
          <div className="w-64 h-2 bg-slate-700 rounded-full overflow-hidden mx-auto">
            <div className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 w-full animate-pulse" />
          </div>
        </div>
      </div>
    );
  }

  // PANTALLA DE BIENVENIDA
  if (screen === 'welcome') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 flex flex-col items-center justify-center p-6">
        <div className="bg-slate-800/50 backdrop-blur-lg rounded-3xl p-8 max-w-md w-full border border-cyan-500/30 shadow-2xl">
          <div className="text-center mb-8">
            <div className="mb-6">
              {CONFIG.LOGO_MAIN ? (
                <img 
                  src={CONFIG.LOGO_MAIN} 
                  alt="CONVALV Logo" 
                  className="w-32 h-32 mx-auto object-contain"
                />
              ) : (
                <div className="bg-gradient-to-r from-cyan-500 to-blue-600 w-32 h-32 rounded-full flex items-center justify-center mx-auto shadow-lg shadow-cyan-500/50">
                  <Heart className="w-16 h-16 text-white" />
                </div>
              )}
            </div>
            <h1 className="text-4xl font-bold text-white mb-2">CONVALV</h1>
            <p className="text-cyan-300 text-sm">Sistema de Diagnóstico Cardiovascular</p>
          </div>
          
          <div className="border-t border-slate-700 pt-6 mb-6">
            <p className="text-slate-300 text-center text-sm">
              Análisis avanzado mediante procesamiento de señales fonocardiográficas e inteligencia artificial
            </p>
          </div>
          
          <button 
            onClick={() => setScreen('capture')}
            className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 text-white py-4 rounded-xl font-semibold hover:shadow-lg hover:shadow-cyan-500/50 transition-all transform hover:scale-105"
          >
            Iniciar Diagnóstico
          </button>
          
          <div className="mt-8 text-center text-xs text-slate-500 space-y-1">
            <div className="flex items-center justify-center gap-2">
              {CONFIG.LOGO_PERSONAL ? (
                <img 
                  src={CONFIG.LOGO_PERSONAL} 
                  alt="Logo Personal" 
                  className="w-8 h-8 object-contain"
                />
              ) : (
                <div className="w-8 h-8 bg-slate-700 rounded-full flex items-center justify-center text-lg">R</div>
              )}
              <span>© Todos los derechos reservados</span>
            </div>
            <div>Creado por: Roberto Martín Gutiérrez</div>
          </div>
        </div>
      </div>
    );
  }

  // PANTALLA DE CAPTURA
  if (screen === 'capture') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-6">
        <div className="max-w-2xl mx-auto">
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-3xl p-8 border border-cyan-500/30">
            <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
              <FileAudio className="text-cyan-400" />
              Captura de Audio Cardíaco
            </h2>
            
            <div className="bg-blue-900/30 border border-blue-500/50 rounded-xl p-6 mb-6">
              <h3 className="text-cyan-300 font-semibold mb-3 flex items-center gap-2">
                <AlertCircle className="w-5 h-5" />
                Instrucciones de Grabación
              </h3>
              <ul className="text-slate-300 text-sm space-y-2">
                <li>• Coloque el micrófono en el pecho, debajo del seno izquierdo (4º espacio intercostal)</li>
                <li>• Mantenga silencio absoluto durante la grabación</li>
                <li>• Contenga la respiración si es posible</li>
                <li>• Evite movimientos al iniciar y finalizar</li>
                <li>• Duración: 20 segundos exactos</li>
              </ul>
            </div>

            <div className="grid gap-4 mb-6">
              <button
                onClick={isRecording ? stopRecording : startRecording}
                className={`p-6 rounded-xl border-2 transition-all ${
                  isRecording 
                    ? 'bg-red-500/20 border-red-500 hover:bg-red-500/30' 
                    : 'bg-cyan-500/10 border-cyan-500/50 hover:bg-cyan-500/20'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className={`w-16 h-16 rounded-full flex items-center justify-center ${
                      isRecording ? 'bg-red-500 animate-pulse' : 'bg-cyan-500'
                    }`}>
                      <Mic className="w-8 h-8 text-white" />
                    </div>
                    <div className="text-left">
                      <div className="text-white font-semibold text-lg">
                        {isRecording ? 'Grabando...' : 'Grabar Audio'}
                      </div>
                      {isRecording && (
                        <div className="text-cyan-300 text-sm">{recordingTime}/20 segundos</div>
                      )}
                    </div>
                  </div>
                  <ChevronRight className="text-cyan-400" />
                </div>
              </button>

              <div className="relative">
                <input
                  type="file"
                  accept="audio/*"
                  onChange={handleFileUpload}
                  className="absolute inset-0 opacity-0 cursor-pointer"
                  id="audio-upload"
                />
                <label
                  htmlFor="audio-upload"
                  className="block p-6 rounded-xl border-2 border-blue-500/50 bg-blue-500/10 hover:bg-blue-500/20 transition-all cursor-pointer"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="w-16 h-16 bg-blue-500 rounded-full flex items-center justify-center">
                        <Upload className="w-8 h-8 text-white" />
                      </div>
                      <div className="text-left">
                        <div className="text-white font-semibold text-lg">Cargar Audio</div>
                        <div className="text-slate-400 text-sm">Mín: 20s | Máx: 30s</div>
                      </div>
                    </div>
                    <ChevronRight className="text-blue-400" />
                  </div>
                </label>
              </div>
            </div>

            <button
              onClick={() => setScreen('welcome')}
              className="w-full bg-slate-700 text-white py-3 rounded-xl hover:bg-slate-600 transition-all"
            >
              Cancelar
            </button>
          </div>
        </div>
      </div>
    );
  }

  // PANTALLA DE PREVIEW
  if (screen === 'preview') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-6 flex items-center">
        <div className="max-w-md mx-auto w-full">
          <div className="bg-slate-800/50 backdrop-blur-lg rounded-3xl p-8 border border-cyan-500/30">
            <h2 className="text-2xl font-bold text-white mb-6 text-center">Vista Previa</h2>
            
            <div className="bg-slate-900/50 rounded-2xl p-8 mb-6 text-center">
              <div className="w-24 h-24 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <FileAudio className="w-12 h-12 text-white" />
              </div>
              
              <audio 
                ref={audioRef} 
                src={audioFile ? URL.createObjectURL(audioFile) : ''} 
                controls  // ← AÑADE ESTA LÍNEA
                className="w-full mb-4"  // ← CAMBIA className
              />
            
                            
              <div className="text-slate-300 text-sm">Toque para reproducir</div>
            </div>

            <div className="grid grid-cols-2 gap-3 mb-6">
              <button
                onClick={runDiagnostic}
                className="bg-gradient-to-r from-green-500 to-emerald-600 text-white py-4 rounded-xl font-semibold hover:shadow-lg hover:shadow-green-500/50 transition-all"
              >
                <CheckCircle className="w-5 h-5 mx-auto mb-1" />
                INICIAR
              </button>
              
              <button
                onClick={() => {
                  setAudioFile(null);
                  setScreen('capture');
                }}
                className="bg-gradient-to-r from-blue-500 to-cyan-600 text-white py-4 rounded-xl font-semibold hover:shadow-lg transition-all"
              >
                <RefreshCw className="w-5 h-5 mx-auto mb-1" />
                CAMBIAR
              </button>
            </div>

            <button
              onClick={() => {
                setAudioFile(null);
                setScreen('welcome');
              }}
              className="w-full bg-red-500/20 border-2 border-red-500 text-red-300 py-3 rounded-xl font-semibold hover:bg-red-500/30 transition-all"
            >
              <X className="w-5 h-5 inline mr-2" />
              DETENER
            </button>
          </div>
        </div>
      </div>
    );
  }

  // PANTALLA DE PROCESAMIENTO
  if (screen === 'processing') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 flex items-center justify-center p-6">
        <div className="text-center">
          <div className="w-32 h-32 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-full flex items-center justify-center mx-auto mb-6 animate-pulse shadow-2xl shadow-cyan-500/50">
            <Heart className="w-16 h-16 text-white" />
          </div>
          <h2 className="text-white text-2xl font-bold mb-4">Analizando Señal Cardíaca</h2>
          <div className="text-cyan-300 mb-6 space-y-2">
            <div>• Extrayendo características espectrales...</div>
            <div>• Detectando eventos valvulares S1/S2...</div>
            <div>• Calculando variabilidad R-R...</div>
            <div>• Ejecutando modelo de IA...</div>
          </div>
          <div className="w-64 h-2 bg-slate-700 rounded-full overflow-hidden mx-auto">
            <div className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 animate-pulse w-full" />
          </div>
        </div>
      </div>
    );
  }

  // PANTALLAS DE MONITORES
  if (screen === 'monitors') {
    const monitors = [
      {
        title: 'Monitor I: Análisis Temporal',
        description: 'Envolvente de energía y sincronía electro-valvular',
        content: (
          <div className="space-y-6">
            {graphs.temporal && (
              <div className="bg-slate-900/80 rounded-xl p-4 mb-4">
                <img 
                  src={graphs.temporal} 
                  alt="Análisis Temporal" 
                  className="w-full h-auto rounded-lg"
                />
              </div>
            )}
            
            {cleanedAudioUrl && (
              <div className="bg-gradient-to-r from-purple-900/40 to-blue-900/40 border-2 border-purple-500/50 rounded-xl p-6 mb-4">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-purple-300 font-semibold flex items-center gap-2">
                    <Activity className="w-5 h-5" />
                    Audio Cardíaco Procesado
                  </h4>
                  <span className="text-xs text-purple-400">Limpio y filtrado</span>
                </div>
                
                <audio 
                    ref={cleanedAudioRef}
                    src={cleanedAudioUrl || ''}  // ← AÑADE src aquí
                    controls
                    className="w-full mb-4"
                    onEnded={() => setIsPlayingCleaned(false)}
                    onError={(e) => {
                        console.error('Error cargando audio limpio:', e);
                        console.log('URL:', cleanedAudioUrl);
                    }}
                    onLoadedData={() => console.log('✅ Audio limpio cargado correctamente')}
                />
                
                <div className="flex items-center gap-4">
                  
                  
                  <div className="flex-1">
                    <div className="text-slate-300 text-sm mb-1">
                      {isPlayingCleaned ? 'Reproduciendo...' : 'Señal fonocardiográfica limpia'}
                    </div>
                    <div className="text-xs text-purple-400">
                      Filtrado 20-400Hz • Sin ruido ambiente
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            <div className="bg-slate-900/80 rounded-xl p-6">
              <h4 className="text-cyan-400 font-semibold mb-4">Detección de Eventos Valvulares</h4>
              <div className="space-y-3 text-sm text-slate-300">
                <div className="flex justify-between"><span>S1 detectados:</span><span className="text-green-400 font-mono">{analysisResults.s1_count}</span></div>
                <div className="flex justify-between"><span>S2 detectados:</span><span className="text-yellow-400 font-mono">{analysisResults.s2_count}</span></div>
                <div className="flex justify-between"><span>Ritmo cardíaco:</span><span className="text-cyan-400 font-mono">{analysisResults.bpm} BPM</span></div>
              </div>
            </div>
            <div className="bg-blue-900/30 border border-blue-500/50 rounded-xl p-4">
              <p className="text-slate-300 text-sm">
                <strong className="text-cyan-300">S1 (verde):</strong> Cierre de válvulas mitral y tricúspide (inicio sístole).
                <br/><strong className="text-yellow-300 mt-2 block">S2 (amarillo):</strong> Cierre de válvulas aórtica y pulmonar (inicio diástole).
              </p>
            </div>
          </div>
        )
      },
      {
        title: 'Monitor II: Análisis Espectral',
        description: 'Densidad de potencia y caracterización de frecuencias',
        content: (
          <div className="space-y-6">
            {graphs.spectral && (
              <div className="bg-slate-900/80 rounded-xl p-4 mb-4">
                <img 
                  src={graphs.spectral} 
                  alt="Análisis Espectral" 
                  className="w-full h-auto rounded-lg"
                />
              </div>
            )}
            
            <div className="bg-slate-900/80 rounded-xl p-6">
              <h4 className="text-cyan-400 font-semibold mb-4">Características Espectrales</h4>
              <div className="space-y-3 text-sm text-slate-300">
                <div className="flex justify-between"><span>Frecuencia dominante:</span><span className="text-green-400 font-mono">{analysisResults.dominant_freq} Hz</span></div>
                <div className="flex justify-between"><span>Entropía de Shannon:</span><span className="text-purple-400 font-mono">{analysisResults.entropy}</span></div>
              </div>
            </div>
            <div className="bg-blue-900/30 border border-blue-500/50 rounded-xl p-4">
              <p className="text-slate-300 text-sm">
                <strong className="text-cyan-300">Espectrograma MEL:</strong> Muestra la distribución de energía en diferentes frecuencias. Patrones anormales pueden indicar soplos o turbulencias.
                <br/><br/><strong className="text-purple-300">Entropía:</strong> Mide la complejidad de la señal. Valores altos (&gt;1.5) sugieren irregularidades.
              </p>
            </div>
          </div>
        )
      },
      {
        title: 'Monitor III: Dinámica Rítmica',
        description: 'Variabilidad R-R y distribución temporal',
        content: (
          <div className="space-y-6">
            {graphs.rhythm && (
              <div className="bg-slate-900/80 rounded-xl p-4 mb-4">
                <img 
                  src={graphs.rhythm} 
                  alt="Dinámica Rítmica" 
                  className="w-full h-auto rounded-lg"
                />
              </div>
            )}
            
            <div className="bg-slate-900/80 rounded-xl p-6">
              <h4 className="text-cyan-400 font-semibold mb-4">Métricas de Variabilidad</h4>
              <div className="space-y-3 text-sm text-slate-300">
                <div className="flex justify-between"><span>Intervalo R-R medio:</span><span className="text-cyan-400 font-mono">{analysisResults.rr_mean} ms</span></div>
                <div className="flex justify-between"><span>Desviación estándar (SDNN):</span><span className="text-yellow-400 font-mono">{analysisResults.rr_std} ms</span></div>
                <div className="flex justify-between"><span>Duración sistólica media:</span><span className="text-green-400 font-mono">{analysisResults.systole_mean} ms</span></div>
                <div className="flex justify-between"><span>Duración diastólica media:</span><span className="text-blue-400 font-mono">{analysisResults.diastole_mean} ms</span></div>
              </div>
            </div>
            <div className="bg-blue-900/30 border border-blue-500/50 rounded-xl p-4">
              <p className="text-slate-300 text-sm">
                <strong className="text-cyan-300">Mapa de Poincaré:</strong> Visualiza la estabilidad del ritmo cardíaco. Puntos dispersos indican arritmia.
                <br/><br/><strong className="text-yellow-300">SDNN:</strong> Variabilidad R-R normal: 20-100 ms. &gt;100 ms puede indicar fibrilación auricular.
              </p>
            </div>
          </div>
        )
      },
      {
        title: 'Monitor IV: Diagnóstico IA',
        description: 'Resultado del modelo ConvalvHolisticNet',
        content: (
          <div className="space-y-6">
            <div className={`rounded-xl p-8 text-center ${
              analysisResults.ia_diagnosis === 'NORMAL' 
                ? 'bg-green-500/20 border-2 border-green-500' 
                : 'bg-red-500/20 border-2 border-red-500'
            }`}>
              <div className="text-6xl font-bold mb-2" style={{
                color: analysisResults.ia_diagnosis === 'NORMAL' ? '#10b981' : '#ef4444'
              }}>
                {analysisResults.ia_diagnosis}
              </div>
              <div className="text-2xl text-white font-semibold">
                Confianza: {analysisResults.ia_confidence}%
              </div>
            </div>
            <div className="bg-blue-900/30 border border-blue-500/50 rounded-xl p-4">
              <p className="text-slate-300 text-sm">
                <strong className="text-cyan-300">Modelo ConvalvHolisticNet V4:</strong> Red neuronal convolucional que combina análisis espectral (espectrograma MEL) con métricas biométricas (BPM, variabilidad R-R, duración sistólica).
                <br/><br/>Entrenado con miles de fonogramas etiquetados por cardiólogos para detectar patologías valvulares, arritmias y otras anomalías.
              </p>
            </div>
          </div>
        )
      }
    ];

    if (currentMonitor < 4) {
      const monitor = monitors[currentMonitor];
      return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-6">
          <div className="max-w-2xl mx-auto">
            <div className="bg-slate-800/50 backdrop-blur-lg rounded-3xl p-6 border border-cyan-500/30">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-white">{monitor.title}</h2>
                <span className="text-cyan-400 text-sm">{currentMonitor + 1}/4</span>
              </div>
              
              <p className="text-slate-400 text-sm mb-6">{monitor.description}</p>
              
              {monitor.content}
              
              <div className="flex gap-3 mt-6">
                {currentMonitor > 0 && (
                  <button
                    onClick={() => setCurrentMonitor(currentMonitor - 1)}
                    className="flex-1 bg-slate-700 text-white py-3 rounded-xl hover:bg-slate-600 transition-all"
                  >
                    Anterior
                  </button>
                )}
                <button
                  onClick={() => setCurrentMonitor(currentMonitor + 1)}
                  className="flex-1 bg-gradient-to-r from-cyan-500 to-blue-600 text-white py-3 rounded-xl font-semibold hover:shadow-lg transition-all"
                >
                  {currentMonitor === 3 ? 'Ver Diagnóstico Final' : 'Siguiente'}
                </button>
              </div>
            </div>
          </div>
        </div>
      );
    } else {
      const finalDx = getFinalDiagnosis();
      return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 p-6">
          <div className="max-w-2xl mx-auto">
            <div className="bg-slate-800/50 backdrop-blur-lg rounded-3xl p-8 border border-cyan-500/30">
              <h2 className="text-3xl font-bold text-white mb-8 text-center">Diagnóstico Final</h2>
              
              <div className={`rounded-2xl p-8 mb-8 text-center border-4 ${
                finalDx.color === 'green' ? 'bg-green-500/20 border-green-500' :
                finalDx.color === 'yellow' ? 'bg-yellow-500/20 border-yellow-500' :
                'bg-red-500/20 border-red-500'
              }`}>
                <div className="text-7xl font-black mb-4" style={{
                  color: finalDx.color === 'green' ? '#10b981' : 
                         finalDx.color === 'yellow' ? '#fbbf24' : '#ef4444'
                }}>
                  {finalDx.status}
                </div>
                {finalDx.color === 'green' ? (
                  <CheckCircle className="w-16 h-16 mx-auto text-green-500" />
                ) : (
                  <AlertCircle className="w-16 h-16 mx-auto" style={{
                    color: finalDx.color === 'yellow' ? '#fbbf24' : '#ef4444'
                  }} />
                )}
              </div>

              <div className="bg-slate-900/80 rounded-xl p-6 mb-6">
                <h3 className="text-cyan-400 font-semibold mb-4 flex items-center gap-2">
                  <Heart className="w-5 h-5" />
                  Resumen de Parámetros
                </h3>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="bg-slate-800/50 p-3 rounded-lg">
                    <div className="text-slate-400 text-xs">Diagnóstico IA</div>
                    <div className="text-white font-mono">{analysisResults.ia_diagnosis}</div>
                  </div>
                  <div className="bg-slate-800/50 p-3 rounded-lg">
                    <div className="text-slate-400 text-xs">Confianza</div>
                    <div className="text-white font-mono">{analysisResults.ia_confidence}%</div>
                  </div>
                  <div className="bg-slate-800/50 p-3 rounded-lg">
                    <div className="text-slate-400 text-xs">BPM</div>
                    <div className="text-white font-mono">{analysisResults.bpm}</div>
                  </div>
                  <div className="bg-slate-800/50 p-3 rounded-lg">
                    <div className="text-slate-400 text-xs">Variabilidad R-R</div>
                    <div className="text-white font-mono">{analysisResults.rr_std} ms</div>
                  </div>
                  <div className="bg-slate-800/50 p-3 rounded-lg">
                    <div className="text-slate-400 text-xs">Entropía</div>
                    <div className="text-white font-mono">{analysisResults.entropy}</div>
                  </div>
                  <div className="bg-slate-800/50 p-3 rounded-lg">
                    <div className="text-slate-400 text-xs">Sístole Media</div>
                    <div className="text-white font-mono">{analysisResults.systole_mean} ms</div>
                  </div>
                </div>
              </div>

              <div className="bg-blue-900/30 border border-blue-500/50 rounded-xl p-6 mb-8">
                <h3 className="text-yellow-300 font-semibold mb-3">Recomendaciones</h3>
                <ul className="text-slate-300 text-sm space-y-2">
                  {finalDx.recommendations.map((rec, idx) => (
                    <li key={idx}>• {rec}</li>
                  ))}
                </ul>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={() => setCurrentMonitor(0)}
                  className="bg-slate-700 text-white py-3 rounded-xl hover:bg-slate-600 transition-all"
                >
                  Ver Monitores
                </button>
                <button
                  onClick={() => {
                    setScreen('welcome');
                    setAudioFile(null);
                    setAnalysisResults(null);
                    setCurrentMonitor(0);
                    setCleanedAudioUrl(null);
                    setIsPlayingCleaned(false);
                  }}
                  className="bg-gradient-to-r from-cyan-500 to-blue-600 text-white py-3 rounded-xl font-semibold hover:shadow-lg transition-all"
                >
                  Nuevo Análisis
                </button>
              </div>
            </div>
          </div>
        </div>
      );
    }
  }

  return null;
};

export default ConvalvApp;