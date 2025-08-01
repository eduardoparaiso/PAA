import numpy as np
from scipy.signal import iirnotch, butter, filtfilt, find_peaks
import wfdb

def load_ECG(path, fs, return_record=False):
  # Lê um registro de ECG no formato WFDB a partir do caminho especificado
  record = wfdb.rdrecord(path)  # Carrega o registro usando a biblioteca WFDB

  # Extrai o sinal analógico do registro (shape: [n_amostras, n_derivações])
  signal = record.p_signal

  # Se return_record for True, retorna também o objeto completo do registro
  if return_record:
    return record, signal
  else:
    # Caso contrário, retorna apenas o sinal
    return signal
    
def highpass_filter(signal, fs=500, cutoff=0.5, order=4):
    # Filtro passa-altas para remover deriva de linha de base
    nyq = 0.5 * fs  # Frequência de Nyquist
    b, a = butter(order, cutoff / nyq, btype='high', analog=False)  # Coeficientes do filtro Butterworth
    return filtfilt(b, a, signal)  # Filtragem com correção de fase

def notch_filter(signal, fs=500, freq=60.0, Q=30):
    # Filtro notch (rejeição de banda) para remover interferência da rede elétrica (60 Hz)
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, signal)

def min_max_scale(signal):
    # Normaliza o sinal para o intervalo [-1, 1]
    return 2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1

def bandpass_filter(signal_data, lowcut=5, highcut=15, fs=500, order=1):
    # Filtro passa-banda para destacar componentes relevantes do ECG (ex.: QRS)
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal_data)
    return y

def derivative(signal_data):
    # Derivada discreta para detectar mudanças rápidas no sinal (bordas do QRS)
    derivative = np.diff(signal_data)
    derivative = np.append(derivative, 0)  # Mantém o mesmo tamanho do sinal original
    return derivative

def square(signal_data):
    # Eleva o sinal ao quadrado para reforçar picos positivos e eliminar valores negativos
    return signal_data ** 2

def moving_window_integration(signal_data, window_size=30):
    # Integração por janela móvel para suavizar e destacar regiões do QRS
    window = np.ones(window_size) / window_size
    integrated = np.convolve(signal_data, window, mode='same')
    return integrated

def detect_qrs(signal_data, fs=500):
    # Pipeline completo para detectar picos QRS:
    filtered = bandpass_filter(signal_data)              # Filtro passa-banda
    deriv = derivative(filtered)                         # Derivada
    squared = square(deriv)                              # Elevação ao quadrado
    integrated = moving_window_integration(squared)      # Integração
    integrated = integrated / np.max(integrated)         # Normalização

    distance = int(0.4 * fs)  # Intervalo mínimo de 400ms entre picos
    peaks, _ = find_peaks(integrated, distance=distance, height=0.5)
    
    return peaks, integrated

def clean_ECG(ECG, canal, fs=500):
    # Pré-processamento completo para um canal do ECG:
    ECG_clean = highpass_filter(ECG[:, canal])  # Remove baixa frequência
    ECG_clean = notch_filter(ECG_clean)         # Remove interferência 60Hz
    ECG_clean = min_max_scale(ECG_clean)        # Normaliza entre [-1, 1]
    return ECG_clean

def extract_beats(ecg_signal, y_signal, qrs_peaks, fs=500, window_size=200):
    # Extrai segmentos de batimentos centrados nos picos QRS
    beats = []
    half_window = window_size // 2

    for peak in qrs_peaks[:5]:  # Extrai apenas os primeiros 5 batimentos detectados
        start = peak - half_window
        end = peak + half_window

        if start < 0 or end > len(ecg_signal):
            continue  # Ignora se a janela ultrapassar os limites do sinal

        beat = ecg_signal[start:end]
        beats.append(beat)

    return beats

def mean_template(segments):
    # Calcula o template médio de batimentos (protótipo)
    # Empilha os segmentos em uma matriz e calcula a média por coluna (tempo)
    return np.mean(np.vstack(segments), axis=0)

def cria_template(df=None, ECG_path=None, canais=[6, 7, 8, 9], ref_template=False):
    # Inicializa listas para armazenar batimentos por canal (pré-definidos como V1–V4)
    batimentos_V1 = []  # Canal 6
    batimentos_V2 = []  # Canal 7
    batimentos_V3 = []  # Canal 8
    batimentos_V4 = []  # Canal 9

    # Itera pelos canais especificados
    for canal in canais:
        if ref_template:
            # Modo: extrair batimentos de múltiplos arquivos referenciados no DataFrame
            for idx, row in df.iterrows():
                ECG = load_ECG(path + row['filename_hr'], 500)  # Carrega o sinal (shape: [n_amostras, n_canais])
                ECG_clean = clean_ECG(ECG, canal)               # Aplica filtragem no canal selecionado
                peaks, _ = detect_qrs(ECG_clean)                 # Detecta picos QRS

                beats = extract_beats(ECG_clean, 0, peaks)       # Extrai batimentos com janela centrada nos picos
                beats = np.array(beats)

                # Verifica se há batimentos extraídos
                if beats.ndim == 2:
                    # Armazena os batimentos no canal correspondente
                    if canal == 6:
                        batimentos_V1.append(beats)
                    elif canal == 7:
                        batimentos_V2.append(beats)
                    elif canal == 8:
                        batimentos_V3.append(beats)
                    elif canal == 9:
                        batimentos_V4.append(beats)

        else:
            # Modo: usar um único arquivo para gerar os batimentos
            ECG = load_ECG(path + ECG_path, 500)         # Carrega o sinal
            ECG_clean = clean_ECG(ECG, canal)            # Filtra o canal
            peaks, _ = detect_qrs(ECG_clean)             # Detecta QRS
            beats = extract_beats(ECG_clean, 0, peaks)   # Extrai batimentos
            beats = np.array(beats)

            if beats.ndim == 2:
                if canal == 6:
                    batimentos_V1.append(beats)
                elif canal == 7:
                    batimentos_V2.append(beats)
                elif canal == 8:
                    batimentos_V3.append(beats)
                elif canal == 9:
                    batimentos_V4.append(beats)

    # Concatena todos os batimentos de cada canal
    batimentos_V1 = np.concatenate(batimentos_V1, axis=0)
    batimentos_V2 = np.concatenate(batimentos_V2, axis=0)
    batimentos_V3 = np.concatenate(batimentos_V3, axis=0)
    batimentos_V4 = np.concatenate(batimentos_V4, axis=0)

    # Calcula o template médio (protótipo) para cada canal
    prototipo = {
        'V1': mean_template(batimentos_V1),
        'V2': mean_template(batimentos_V2),
        'V3': mean_template(batimentos_V3),
        'V4': mean_template(batimentos_V4)
    }

    return prototipo