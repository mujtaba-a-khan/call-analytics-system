"""
Audio Processing Module

Handles audio file conversion, normalization, and preparation
for speech-to-text processing. Ensures all audio is in the
correct format for Whisper STT.
"""

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
from pydub import AudioSegment
from pydub.utils import which

# Configure logging for this module
logger = logging.getLogger(__name__)

# Ensure ffmpeg is available for audio processing
ffmpeg_path = which("ffmpeg")
if not ffmpeg_path:
    logger.warning("FFmpeg not found. Audio processing may fail.")
else:
    AudioSegment.converter = ffmpeg_path


class AudioProcessor:
    """
    Handles all audio file processing operations including
    format conversion, normalization, and caching.
    """

    def __init__(self, config: dict):
        """
        Initialize the audio processor with configuration.

        Args:
            config: Dictionary containing audio processing settings
        """
        self.sample_rate = config.get("sample_rate", 16000)
        self.channels = config.get("channels", 1)
        self.bit_depth = config.get("bit_depth", 16)
        self.cache_dir = Path(config.get("cache_dir", "data/cache"))
        self.output_dir = Path(config.get("processed_dir", "data/processed"))

        # Create necessary directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"AudioProcessor initialized with sample_rate={self.sample_rate}")

    def process_audio_file(self, file_path: Path, use_cache: bool = True) -> tuple[Path, float]:
        """
        Process a single audio file by converting it to the standard format.

        Args:
            file_path: Path to the input audio file
            use_cache: Whether to use cached processed files

        Returns:
            Tuple of (processed_file_path, duration_seconds)
        """
        # Generate cache key based on file content and processing parameters
        cache_key = self._generate_cache_key(file_path)
        processed_path = self.output_dir / f"{cache_key}.wav"

        # Check if we have a cached processed version
        if use_cache and processed_path.exists():
            logger.info(f"Using cached processed audio: {processed_path}")
            duration = self._get_audio_duration(processed_path)
            return processed_path, duration

        try:
            # Load the audio file
            logger.info(f"Processing audio file: {file_path}")
            audio = AudioSegment.from_file(str(file_path))

            # Apply normalization settings
            audio = self._normalize_audio(audio)

            # Export the processed audio
            audio.export(
                str(processed_path), format="wav", parameters=["-acodec", "pcm_s16le"]  # 16-bit PCM
            )

            duration = len(audio) / 1000.0  # Convert from milliseconds to seconds

            # Save metadata for this processed file
            self._save_processing_metadata(file_path, processed_path, duration)

            logger.info(f"Audio processed successfully: {processed_path}")
            return processed_path, duration

        except Exception as e:
            logger.error(f"Error processing audio file {file_path}: {e}")
            raise AudioProcessingError(f"Failed to process audio: {e}") from e

    def process_audio(self, file_path: Path, use_cache: bool = True) -> Path:
        """Compatibility wrapper returning only the processed file path."""
        processed_path, _ = self.process_audio_file(file_path, use_cache=use_cache)
        return processed_path

    def _normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        """
        Normalize audio to standard format for STT processing.

        Args:
            audio: Input audio segment

        Returns:
            Normalized audio segment
        """
        # Convert to mono if needed
        if audio.channels > 1:
            audio = audio.set_channels(self.channels)
            logger.debug("Converted audio to mono")

        # Set sample rate
        if audio.frame_rate != self.sample_rate:
            audio = audio.set_frame_rate(self.sample_rate)
            logger.debug(f"Resampled audio to {self.sample_rate} Hz")

        # Set sample width (bit depth)
        target_sample_width = self.bit_depth // 8
        if audio.sample_width != target_sample_width:
            audio = audio.set_sample_width(target_sample_width)
            logger.debug(f"Set sample width to {self.bit_depth} bits")

        # Apply normalization to prevent clipping
        audio = self._normalize_volume(audio)

        return audio

    def _normalize_volume(
        self,
        audio: AudioSegment,
        target_dbfs: float = -20.0,
    ) -> AudioSegment:
        """
        Normalize audio volume to a target level.

        Args:
            audio: Input audio segment
            target_dbfs: Target volume in dBFS (decibels relative to full scale)

        Returns:
            Volume-normalized audio
        """
        current_dbfs = audio.dBFS
        change_in_dbfs = target_dbfs - current_dbfs

        # Only normalize if the change is significant
        if abs(change_in_dbfs) > 1.0:
            audio = audio.apply_gain(change_in_dbfs)
            logger.debug("Applied volume normalization: %.1f dB", change_in_dbfs)

        return audio

    def _generate_cache_key(self, file_path: Path) -> str:
        """
        Generate a unique cache key for the file and processing parameters.

        Args:
            file_path: Path to the input file

        Returns:
            Hexadecimal cache key string
        """
        # Create a hash based on file content and processing parameters
        hasher = hashlib.sha256()

        # Include file statistics
        stat = file_path.stat()
        hasher.update(str(stat.st_size).encode())
        hasher.update(str(stat.st_mtime_ns).encode())

        # Include processing parameters
        params = f"{self.sample_rate}_{self.channels}_{self.bit_depth}"
        hasher.update(params.encode())

        # Include file name for readability
        hasher.update(file_path.name.encode())

        return hasher.hexdigest()[:16]  # Use first 16 characters

    def _get_audio_duration(self, file_path: Path) -> float:
        """
        Get the duration of an audio file in seconds.

        Args:
            file_path: Path to the audio file

        Returns:
            Duration in seconds
        """
        try:
            audio = AudioSegment.from_file(str(file_path))
            return len(audio) / 1000.0
        except Exception as e:
            logger.error(f"Error getting audio duration: {e}")
            return 0.0

    def _save_processing_metadata(self, original_path: Path, processed_path: Path, duration: float):
        """
        Save metadata about the processed audio file.

        Args:
            original_path: Path to the original audio file
            processed_path: Path to the processed audio file
            duration: Duration of the audio in seconds
        """
        metadata = {
            "original_file": str(original_path),
            "processed_file": str(processed_path),
            "duration_seconds": duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "processing_timestamp": pd.Timestamp.now().isoformat(),
        }

        metadata_path = processed_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Saved processing metadata to {metadata_path}")

    def batch_process_audio_files(self, file_paths: list[Path]) -> list[tuple[Path, float]]:
        """
        Process multiple audio files in batch.

        Args:
            file_paths: List of paths to audio files

        Returns:
            List of tuples containing (processed_path, duration)
        """
        results = []
        total_files = len(file_paths)

        for idx, file_path in enumerate(file_paths, 1):
            logger.info(f"Processing file {idx}/{total_files}: {file_path.name}")

            try:
                processed_path, duration = self.process_audio_file(file_path)
                results.append((processed_path, duration))
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                # Continue processing other files
                continue

        logger.info(f"Batch processing complete: {len(results)}/{total_files} successful")
        return results

    def validate_audio_file(self, file_path: Path) -> bool:
        """
        Validate that a file is a processable audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            True if file is valid, False otherwise
        """
        # Check if file exists
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False

        # Check file size (not empty, not too large)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb == 0:
            logger.error(f"File is empty: {file_path}")
            return False
        if file_size_mb > 500:  # Max 500 MB
            logger.error(f"File too large ({file_size_mb:.1f} MB): {file_path}")
            return False

        # Try to load the file to verify it's valid audio
        try:
            audio = AudioSegment.from_file(str(file_path))
            duration_minutes = len(audio) / 60000.0

            # Check duration limits
            if duration_minutes > 60:  # Max 60 minutes
                logger.error(f"Audio too long ({duration_minutes:.1f} min): {file_path}")
                return False

            return True

        except Exception as e:
            logger.error(f"Invalid audio file {file_path}: {e}")
            return False

    def clear_cache(self) -> int:
        """
        Clear the audio processing cache.

        Returns:
            Number of files removed
        """
        count = 0
        for file_path in self.cache_dir.glob("*"):
            try:
                file_path.unlink()
                count += 1
            except Exception as e:
                logger.error(f"Failed to remove cache file {file_path}: {e}")

        logger.info(f"Cleared {count} files from cache")
        return count


class AudioProcessingError(Exception):
    """Custom exception for audio processing errors"""

    pass
