import os

from piper import PiperVoice
import wave
import numpy as np
import soundfile as sf
import io

from core.providers.tts.base import TTSProviderBase
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)

        self.model_dir = config.get("model_dir")
        self.output_dir = config.get("output_dir")
        os.makedirs(self.output_dir, exist_ok=True)

        logger.bind(tag=TAG).info(f"Loading PiperTTS-fin model from {self.model_dir}")
        self.voice = PiperVoice.load(self.model_dir + "/fi_FI-harri-medium.onnx")

    async def text_to_speak(self, text, output_file):
        """Convert text to speech and save to a file."""
        try:
            # Generate speech using the PiperTTS-fin model
            logger.bind(tag=TAG).info(f"Generating speech for text: {text}")
            audio_bytes_raw = b''.join([chunk.audio_int16_bytes for chunk in self.voice.synthesize(text)])

            # Convert raw 16-bit PCM bytes to NumPy float32
            samples = np.frombuffer(audio_bytes_raw, dtype=np.int16).astype(np.float32) / 32768.0
            wav_bytes_io = io.BytesIO()
            sf.write(wav_bytes_io, samples, samplerate=self.voice.config.sample_rate, format='WAV')
            audio_bytes = wav_bytes_io.getvalue()

            # Save the generated audio to a file
            if output_file:
                with wave.open(output_file, "wb") as wav_file:
                    self.voice.synthesize_wav(text, wav_file)
                logger.bind(tag=TAG).info(f"Audio saved to {output_file}")
            else:
                return audio_bytes

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error during text-to-speech: {e}")
            raise Exception(f"{__name__} error: {e}")