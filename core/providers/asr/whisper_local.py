import time
import os
import sys
import io
import psutil
from config.logger import setup_logging
from typing import Optional, Tuple, List
from core.providers.asr.base import ASRProviderBase
import shutil
from core.providers.asr.dto.dto import InterfaceType
from transformers import pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import wave

TAG = __name__
logger = setup_logging()

MAX_RETRIES = 2
RETRY_DELAY = 1  # 重试延迟（秒）

# 捕获标准输出
class CaptureOutput:
    def __enter__(self):
        self._output = io.StringIO()
        self._original_stdout = sys.stdout
        sys.stdout = self._output

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        self.output = self._output.getvalue()
        self._output.close()

        # 将捕获到的内容通过 logger 输出
        if self.output:
            logger.bind(tag=TAG).info(self.output.strip())

class ASRProvider(ASRProviderBase):
    def __init__(self, config: dict, delete_audio_file: bool):
        super().__init__()
        
        # Check memory requirements
        min_mem_bytes = 2 * 1024 * 1024 * 1024
        total_mem = psutil.virtual_memory().total
        if total_mem < min_mem_bytes:
            logger.bind(tag=TAG).error(f"可用内存不足2G，当前仅有 {total_mem / (1024*1024):.2f} MB，可能无法启动ASR")

        self.interface_type = InterfaceType.LOCAL
        self.model_dir = config.get("model_dir")
        self.output_dir = config.get("output_dir")
        self.delete_audio_file = delete_audio_file

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Load the processor and model
        processor = WhisperProcessor.from_pretrained(self.model_dir)
        model = WhisperForConditionalGeneration.from_pretrained(self.model_dir)

        # Set the task and language in the model configuration
        model.config.task = "transcribe"  # Set the task to transcribe
        model.config.language = "fi"  # Set the language to Finnish

        '''
        # Force the model to use Finnish
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="finnish", task="transcribe")
        model.config.forced_decoder_ids = forced_decoder_ids
        '''
        # Load the Whisper model locally
        with CaptureOutput():
            #self.asr_pipeline = pipeline("automatic-speech-recognition", model=self.model_dir)
            self.asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,  # Explicitly pass the tokenizer
                feature_extractor=processor.feature_extractor  # Pass the feature extractor
            )

    def pcm_to_wav(self, pcm_data: bytes, output_path: str, sample_rate: int = 16000, channels: int = 1):
        """Convert PCM data to WAV format."""
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(channels)  # Mono audio
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(sample_rate)  # Sample rate
            wav_file.writeframes(pcm_data)

    async def speech_to_text(
        self, opus_data: List[bytes], session_id: str, audio_format="opus"
    ) -> Tuple[Optional[str], Optional[str]]:
        """Speech-to-text processing logic"""
        file_path = None
        retry_count = 0

        while retry_count < MAX_RETRIES:
            try:
                # Combine all opus data packets
                if audio_format == "pcm":
                    pcm_data = opus_data
                else:
                    pcm_data = self.decode_opus(opus_data)

                combined_pcm_data = b"".join(pcm_data)

                # Check disk space
                if not self.delete_audio_file:
                    free_space = shutil.disk_usage(self.output_dir).free
                    if free_space < len(combined_pcm_data) * 2:  # Reserve 2x space
                        raise OSError("Insufficient disk space")

                # Save as WAV file if needed
                if self.delete_audio_file:
                    pass
                else:
                    file_path = self.save_audio_to_file(pcm_data, session_id)

                # Convert PCM to WAV
                wav_path = os.path.join(self.output_dir, f"{session_id}.wav")
                self.pcm_to_wav(combined_pcm_data, wav_path)

                # Perform speech recognition
                start_time = time.time()
                transcription = self.asr_pipeline(wav_path)
                text = transcription["text"]
                logger.bind(tag=TAG).debug(
                    f"Speech recognition took: {time.time() - start_time:.3f}s | Result: {text}"
                )

                return text, wav_path

            except OSError as e:
                retry_count += 1
                if retry_count >= MAX_RETRIES:
                    logger.bind(tag=TAG).error(
                        f"Speech recognition failed (retried {retry_count} times): {e}", exc_info=True
                    )
                    return "", file_path
                logger.bind(tag=TAG).warning(
                    f"Speech recognition failed, retrying ({retry_count}/{MAX_RETRIES}): {e}"
                )
                time.sleep(RETRY_DELAY)

            except Exception as e:
                logger.bind(tag=TAG).error(f"Speech recognition failed: {e}", exc_info=True)
                return "", file_path

            finally:
                # File cleanup logic
                if self.delete_audio_file and file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.bind(tag=TAG).debug(f"Deleted temporary audio file: {file_path}")
                    except Exception as e:
                        logger.bind(tag=TAG).error(
                            f"Failed to delete file: {file_path} | Error: {e}"
                        )