import numpy as np
import onnxruntime as ort
from datetime import datetime, timezone
import soundfile as sf
import time

def get_audio_in_chunks():

    # Lê o áudio
    audio, sr = sf.read("test-1-16k.wav")

    # Converte para 16kHz mono float32
    if sr != 16000:
        raise ValueError("Áudio deve estar em 16kHz para Silero VAD")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # mix stereo -> mono
    audio = audio.astype(np.float32)

    # Get chunks
    window_size = 512
    chunks = [audio[i:i+window_size] for i in range(0, len(audio), window_size)]
    return chunks

def process_chunks(chunks, window_size = 512):
    voice_detected_stamps = []

    # Inicializa estados ocultos zerados
    h = np.zeros((2, 1, 64), dtype=np.float32)
    c = np.zeros((2, 1, 64), dtype=np.float32)
    sr = np.array(16000, dtype=np.int64)

    counter = 0
    for chunk in chunks:
        if len(chunk) < window_size:
            pad = np.zeros(window_size - len(chunk), dtype=np.float32)
            chunk = np.concatenate([chunk, pad])
        
        state = np.zeros((2, 1, 128), dtype=np.float32)
        ort_inputs = {
            "input": chunk.reshape(1, -1),
            "state": state,
            "sr": sr
        }
        ort_outs = session.run(None, ort_inputs)
        
        out, state = ort_outs  # out é probabilidade de "voz"
        if out[0][0] > 0.5:
            counter += 1
            # print("chunk", chunk)
        
    return counter


# Carrega o modelo Silero VAD em formato ONNX
session = ort.InferenceSession("silero_vad.onnx")

start = time.perf_counter()

chunks = get_audio_in_chunks()

counter = process_chunks(chunks)

end = time.perf_counter()

print("session up", len(chunks))
print("Voice detection counter", counter)
print(f"Tempo de execução: {end - start:.4f} segundos")