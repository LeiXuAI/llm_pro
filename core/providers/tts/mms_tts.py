import os
import torch
from scipy.io.wavfile import write as write_wav
import numpy as np
import io
import base64
from transformers import VitsModel, AutoTokenizer
from core.providers.tts.base import TTSProviderBase
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)

        # Load the model and tokenizer from the local directory or Hugging Face hub
        self.model_dir = config.get("model_dir")
        self.output_dir = config.get("output_dir")
        os.makedirs(self.output_dir, exist_ok=True)

        logger.bind(tag=TAG).info(f"Loading MMS-TTS model from {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = VitsModel.from_pretrained(self.model_dir)

    async def text_to_speak(self, text, output_file):
        """Convert text to speech and save to a file."""
        try:
            # Tokenize the input text
            logger.bind(tag=TAG).info(f"Tokenizing text: {text}")
            inputs = self.tokenizer(text, return_tensors="pt")

            # Generate speech using the VitsModel
            logger.bind(tag=TAG).info(f"Generating speech for text: {text}")
      
            with torch.no_grad():
                outputs = self.model(**inputs)

            waveform = outputs.waveform.squeeze().cpu().numpy()
            buffer = io.BytesIO()

            write_wav(buffer, rate=self.model.config.sampling_rate, data=(waveform * 32767).astype(np.int16))
            audio_bytes = buffer.getvalue()
            buffer.close()

            # Save the generated audio to a file
            if output_file:
                with open(output_file, "wb") as f:
                        f.write(audio_bytes)
                logger.bind(tag=TAG).info(f"Audio saved to {output_file}")
            else:
                return audio_bytes

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error during text-to-speech: {e}")
            raise Exception(f"{__name__} error: {e}")