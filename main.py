def get_audio():
    import soundfile as sf
    import numpy as np

    # Lê o áudio
    audio, sr = sf.read("test-1.wav")

    # Converte para 16kHz mono float32
    if sr != 16000:
        raise ValueError("Áudio deve estar em 16kHz para Silero VAD")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # mix stereo -> mono
    audio = audio.astype(np.float32)
    return audio

def wav_to_text(audio):
    # Divide em blocos de 512 samples (~32ms)
    window_size = 512
    chunks = [audio[i:i+window_size] for i in range(0, len(audio), window_size)]