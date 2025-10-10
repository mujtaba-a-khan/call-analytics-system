"""
Whisper Speech-to-Text Module

Implements offline speech-to-text functionality using faster-whisper.
Includes caching, batch processing, and confidence scoring.
"""

import hashlib
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("faster-whisper not installed. STT functionality will be limited.")

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Data class for transcription results"""
    transcript: str
    segments: list[dict]
    duration_seconds: float
    language: str
    confidence: float
    model_used: str

    def to_dict(self) -> dict[str, Any]:
        """Provide a dict view for legacy code expecting mapping semantics."""
        return {
            'text': self.transcript,
            'transcript': self.transcript,
            'segments': self.segments,
            'duration': self.duration_seconds,
            'duration_seconds': self.duration_seconds,
            'language': self.language,
            'confidence': self.confidence,
            'model': self.model_used,
            'model_used': self.model_used,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Compatibility helper matching dict.get signature."""
        return self.to_dict().get(key, default)


class WhisperSTT:
    """
    Speech-to-text engine using Whisper model.
    Provides efficient transcription with caching support.
    """

    def __init__(self, config: dict):
        """
        Initialize the Whisper STT engine.

        Args:
            config: Configuration dictionary with model settings
        """
        if not WHISPER_AVAILABLE:
            raise RuntimeError("faster-whisper is not installed. Please install it first.")

        self.model_size = config.get('model_size', 'small.en')
        self.compute_type = config.get('compute_type', 'int8')
        self.device = config.get('device', 'auto')
        self.num_workers = config.get('num_workers', 1)
        self.beam_size = config.get('beam_size', 1)
        self.temperature = config.get('temperature', 0.0)
        self.vad_filter = config.get('vad_filter', False)
        self.default_language = config.get('language')

        # Cache settings
        self.cache_dir = Path(config.get('cache_dir', 'data/cache/stt'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the model
        self.model = None
        self._load_model()

        logger.info(f"WhisperSTT initialized with model: {self.model_size}")

    def _load_model(self):
        """Load the Whisper model into memory"""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")

            # Determine device automatically if set to 'auto'
            if self.device == 'auto':
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = self.device

            self.model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=self.compute_type,
                num_workers=self.num_workers
            )

            logger.info(f"Model loaded successfully on device: {device}")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(f"Could not load Whisper model: {e}") from e

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        use_cache: bool = True
    ) -> TranscriptionResult:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file (should be WAV 16kHz mono)
            language: Preferred language code (None uses defaults)
            use_cache: Whether to use cached transcriptions

        Returns:
            TranscriptionResult containing transcript and metadata
        """
        # Check cache first
        if use_cache and language is None:
            cached_result = self._load_from_cache(audio_path)
            if cached_result:
                logger.info(f"Using cached transcription for {audio_path.name}")
                return cached_result

        try:
            logger.info(f"Transcribing audio: {audio_path.name}")

            language_hint = language or self.default_language
            # Use english for .en models when no explicit language supplied
            if language_hint is None and self.model_size.endswith('.en'):
                language_hint = 'en'

            # Perform transcription
            segments, info = self.model.transcribe(
                str(audio_path),
                language=language_hint,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                temperature=self.temperature
            )

            # Process segments into structured format
            segment_list = []
            full_text_parts = []
            total_confidence = 0.0
            segment_count = 0

            for segment in segments:
                text = segment.text.strip()
                if text:
                    full_text_parts.append(text)
                    segment_dict = {
                        'start': float(segment.start),
                        'end': float(segment.end),
                        'text': text,
                        'confidence': float(getattr(segment, 'confidence', 0.95))
                    }
                    segment_list.append(segment_dict)
                    total_confidence += segment_dict['confidence']
                    segment_count += 1

            # Combine all segments into full transcript
            full_transcript = ' '.join(full_text_parts).strip()

            # Calculate average confidence
            avg_confidence = total_confidence / segment_count if segment_count > 0 else 0.0

            # Get audio duration from info
            duration = float(getattr(info, 'duration', 0.0))

            detected_language = language_hint or getattr(info, 'language', None) or 'unknown'

            # Create result object
            result = TranscriptionResult(
                transcript=full_transcript,
                segments=segment_list,
                duration_seconds=duration,
                language=detected_language,
                confidence=avg_confidence,
                model_used=self.model_size
            )

            # Save to cache only when we can safely reuse without language-specific differences
            if use_cache and language is None:
                self._save_to_cache(audio_path, result)

            logger.info(f"Transcription complete: {len(full_transcript)} chars, "
                       f"{segment_count} segments, {avg_confidence:.2f} confidence")

            return result

        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            raise TranscriptionError(f"Failed to transcribe audio: {e}") from e

    def batch_transcribe(self, audio_paths: list[Path]) -> list[TranscriptionResult]:
        """
        Transcribe multiple audio files in batch.

        Args:
            audio_paths: List of paths to audio files

        Returns:
            List of TranscriptionResult objects
        """
        results = []
        total_files = len(audio_paths)

        for idx, audio_path in enumerate(audio_paths, 1):
            logger.info(f"Transcribing {idx}/{total_files}: {audio_path.name}")

            try:
                result = self.transcribe(audio_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_path}: {e}")
                # Add empty result to maintain order
                results.append(TranscriptionResult(
                    transcript="",
                    segments=[],
                    duration_seconds=0.0,
                    language='en',
                    confidence=0.0,
                    model_used=self.model_size
                ))

        logger.info(f"Batch transcription complete: {len(results)} files processed")
        return results

    def _generate_cache_key(self, audio_path: Path) -> str:
        """
        Generate a unique cache key for the audio file and model settings.

        Args:
            audio_path: Path to the audio file

        Returns:
            Hexadecimal cache key string
        """
        hasher = hashlib.sha256()

        # Include file information
        stat = audio_path.stat()
        hasher.update(str(stat.st_size).encode())
        hasher.update(str(stat.st_mtime_ns).encode())
        hasher.update(audio_path.name.encode())

        # Include model settings
        settings = f"{self.model_size}_{self.compute_type}_{self.beam_size}"
        hasher.update(settings.encode())

        return hasher.hexdigest()[:16]

    def _save_to_cache(self, audio_path: Path, result: TranscriptionResult):
        """
        Save transcription result to cache.

        Args:
            audio_path: Path to the original audio file
            result: TranscriptionResult to cache
        """
        cache_key = self._generate_cache_key(audio_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            logger.debug(f"Saved transcription to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _load_from_cache(self, audio_path: Path) -> TranscriptionResult | None:
        """
        Load transcription result from cache if available.

        Args:
            audio_path: Path to the original audio file

        Returns:
            Cached TranscriptionResult or None if not found
        """
        cache_key = self._generate_cache_key(audio_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                return result
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                # Remove corrupted cache file
                cache_file.unlink(missing_ok=True)

        return None

    def clear_cache(self) -> int:
        """
        Clear the transcription cache.

        Returns:
            Number of cache files removed
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.error(f"Failed to remove cache file {cache_file}: {e}")

        logger.info(f"Cleared {count} transcription cache files")
        return count

    def validate_transcript(self, transcript: str, min_tokens: int = 20) -> bool:
        """
        Validate that a transcript meets minimum quality standards.

        Args:
            transcript: The transcript text to validate
            min_tokens: Minimum number of tokens required

        Returns:
            True if transcript is valid, False otherwise
        """
        if not transcript or not transcript.strip():
            return False

        # Simple token counting (split by whitespace)
        tokens = transcript.strip().split()

        if len(tokens) < min_tokens:
            logger.debug(f"Transcript too short: {len(tokens)} tokens < {min_tokens}")
            return False

        # Check for excessive repetition (possible transcription error)
        unique_tokens = set(tokens)
        if len(unique_tokens) < len(tokens) * 0.2:  # Less than 20% unique
            logger.debug("Transcript has excessive repetition")
            return False

        return True


class TranscriptionError(Exception):
    """Custom exception for transcription errors"""
    pass
