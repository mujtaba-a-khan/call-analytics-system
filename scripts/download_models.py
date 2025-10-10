"""
Model Download Script for Call Analytics System

This script downloads and sets up required ML models including
Whisper for speech-to-text and sentence transformers for embeddings.
"""

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import get_logger, setup_logging

# Model configurations
MODELS_CONFIG = {
    "whisper": {
        "models": {
            "tiny": {
                "url": (
                    "https://openaipublic.azureedge.net/main/whisper/models/"
                    "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
                ),
                "size": "39 MB",
                "sha256": "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9",
            },
            "base": {
                "url": (
                    "https://openaipublic.azureedge.net/main/whisper/models/"
                    "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt"
                ),
                "size": "74 MB",
                "sha256": "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e",
            },
            "small": {
                "url": (
                    "https://openaipublic.azureedge.net/main/whisper/models/"
                    "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt"
                ),
                "size": "244 MB",
                "sha256": "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794",
            },
            "medium": {
                "url": (
                    "https://openaipublic.azureedge.net/main/whisper/models/"
                    "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt"
                ),
                "size": "769 MB",
                "sha256": "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1",
            },
            "large": {
                "url": (
                    "https://openaipublic.azureedge.net/main/whisper/models/"
                    "81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt"
                ),
                "size": "1550 MB",
                "sha256": "81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524",
            },
        }
    },
    "sentence_transformers": {
        "models": [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "multi-qa-MiniLM-L6-cos-v1",
            "paraphrase-multilingual-MiniLM-L12-v2",
        ]
    },
    "ollama": {"models": ["llama3", "mistral", "nomic-embed-text"]},
}


class ModelDownloader:
    """
    Handles downloading and setup of ML models for the call analytics system.
    """

    def __init__(self, models_dir: Path, logger: logging.Logger):
        """
        Initialize model downloader.

        Args:
            models_dir: Directory to store downloaded models
            logger: Logger instance
        """
        self.models_dir = models_dir
        self.logger = logger
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def download_file(
        self,
        url: str,
        destination: Path,
        expected_sha256: str | None = None,
    ) -> bool:
        """
        Download a file with progress bar and optional hash verification.

        Args:
            url: URL to download from
            destination: Local path to save file
            expected_sha256: Expected SHA256 hash for verification

        Returns:
            True if download successful, False otherwise
        """
        try:
            self.logger.info(f"Downloading {url} to {destination}")

            # Make request
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Get total file size
            total_size = int(response.headers.get("content-length", 0))

            # Download with progress bar
            with (
                open(destination, "wb") as file,
                tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=destination.name,
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    pbar.update(len(chunk))

            # Verify hash if provided
            if expected_sha256:
                self.logger.info("Verifying file hash...")
                if not self.verify_file_hash(destination, expected_sha256):
                    self.logger.error("Hash verification failed!")
                    destination.unlink()  # Remove corrupted file
                    return False

            self.logger.info(f"Successfully downloaded {destination.name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to download {url}: {e}")
            return False

    def verify_file_hash(self, file_path: Path, expected_sha256: str) -> bool:
        """
        Verify file SHA256 hash.

        Args:
            file_path: Path to file
            expected_sha256: Expected hash

        Returns:
            True if hash matches, False otherwise
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        actual_hash = sha256_hash.hexdigest()
        return actual_hash == expected_sha256

    def download_whisper_models(self, model_sizes: list[str]) -> None:
        """
        Download Whisper STT models.

        Args:
            model_sizes: List of model sizes to download
        """
        whisper_dir = self.models_dir / "whisper"
        whisper_dir.mkdir(exist_ok=True)

        for size in model_sizes:
            if size not in MODELS_CONFIG["whisper"]["models"]:
                self.logger.warning(f"Unknown Whisper model size: {size}")
                continue

            model_info = MODELS_CONFIG["whisper"]["models"][size]
            model_path = whisper_dir / f"{size}.pt"

            # Check if already exists
            if model_path.exists():
                self.logger.info(f"Whisper {size} model already exists, skipping")
                continue

            # Download model
            self.logger.info(f"Downloading Whisper {size} model ({model_info['size']})...")
            success = self.download_file(model_info["url"], model_path, model_info.get("sha256"))

            if not success:
                self.logger.error(f"Failed to download Whisper {size} model")

    def download_sentence_transformers(self, model_names: list[str]) -> None:
        """
        Download sentence transformer models using the library's built-in downloader.

        Args:
            model_names: List of model names to download
        """
        try:
            from sentence_transformers import SentenceTransformer

            st_dir = self.models_dir / "sentence_transformers"
            st_dir.mkdir(exist_ok=True)

            for model_name in model_names:
                self.logger.info(f"Downloading sentence transformer: {model_name}")

                try:
                    # Download and cache model
                    model = SentenceTransformer(model_name, cache_folder=str(st_dir))
                    self.logger.info(f"Successfully downloaded {model_name}")

                    # Save model info
                    info_path = st_dir / f"{model_name}_info.json"
                    with open(info_path, "w") as f:
                        json.dump(
                            {
                                "name": model_name,
                                "dimension": model.get_sentence_embedding_dimension(),
                                "max_seq_length": model.max_seq_length,
                            },
                            f,
                            indent=2,
                        )

                except Exception as e:
                    self.logger.error(f"Failed to download {model_name}: {e}")

        except ImportError:
            self.logger.error(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )

    def setup_ollama_models(self, model_names: list[str]) -> None:
        """
        Pull Ollama models if Ollama is installed.

        Args:
            model_names: List of Ollama model names
        """
        # Check if Ollama is installed
        try:
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.warning("Ollama not installed or not in PATH")
                return
        except FileNotFoundError:
            self.logger.warning("Ollama not installed. Visit https://ollama.ai for installation")
            return

        for model_name in model_names:
            self.logger.info(f"Pulling Ollama model: {model_name}")

            try:
                # Pull model using Ollama CLI
                result = subprocess.run(
                    ["ollama", "pull", model_name], capture_output=True, text=True
                )

                if result.returncode == 0:
                    self.logger.info(f"Successfully pulled {model_name}")
                else:
                    self.logger.error(f"Failed to pull {model_name}: {result.stderr}")

            except Exception as e:
                self.logger.error(f"Error pulling {model_name}: {e}")

    def create_model_registry(self) -> None:
        """
        Create a registry file with information about all downloaded models.
        """
        registry = {"whisper": {}, "sentence_transformers": {}, "ollama": {}}

        # Check Whisper models
        whisper_dir = self.models_dir / "whisper"
        if whisper_dir.exists():
            for model_file in whisper_dir.glob("*.pt"):
                size = model_file.stem
                registry["whisper"][size] = {
                    "path": str(model_file),
                    "size": model_file.stat().st_size,
                }

        # Check sentence transformer models
        st_dir = self.models_dir / "sentence_transformers"
        if st_dir.exists():
            for info_file in st_dir.glob("*_info.json"):
                with open(info_file) as f:
                    info = json.load(f)
                    registry["sentence_transformers"][info["name"]] = info

        # Check Ollama models
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)

            if result.returncode == 0:
                # Parse Ollama list output
                lines = result.stdout.strip().split("\n")[1:]  # Skip header
                for line in lines:
                    if line:
                        parts = line.split()
                        if parts:
                            model_name = parts[0]
                            registry["ollama"][model_name] = {"installed": True}
        except Exception:
            pass

        # Save registry
        registry_path = self.models_dir / "model_registry.json"
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

        self.logger.info(f"Model registry saved to {registry_path}")

    def verify_installations(self) -> dict[str, bool]:
        """
        Verify that required models are installed.

        Returns:
            Dictionary of model types and their installation status
        """
        status = {}

        # Check Whisper
        whisper_dir = self.models_dir / "whisper"
        status["whisper"] = whisper_dir.exists() and any(whisper_dir.glob("*.pt"))

        # Check sentence transformers
        st_dir = self.models_dir / "sentence_transformers"
        status["sentence_transformers"] = st_dir.exists() and any(st_dir.glob("*_info.json"))

        # Check Ollama
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True)
            status["ollama"] = result.returncode == 0
        except Exception:
            status["ollama"] = False

        return status


def main():
    """
    Main function to run the model download script.
    """
    parser = argparse.ArgumentParser(
        description="Download and setup ML models for Call Analytics System"
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory to store models (default: models/)",
    )

    parser.add_argument(
        "--whisper-sizes",
        nargs="+",
        default=["small"],
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model sizes to download (default: small)",
    )

    parser.add_argument(
        "--sentence-transformers",
        nargs="+",
        default=["all-MiniLM-L6-v2"],
        help="Sentence transformer models to download",
    )

    parser.add_argument(
        "--ollama-models", nargs="+", default=["nomic-embed-text"], help="Ollama models to pull"
    )

    parser.add_argument(
        "--skip-whisper", action="store_true", help="Skip downloading Whisper models"
    )

    parser.add_argument(
        "--skip-sentence-transformers",
        action="store_true",
        help="Skip downloading sentence transformer models",
    )

    parser.add_argument("--skip-ollama", action="store_true", help="Skip setting up Ollama models")

    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify existing installations"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level="INFO", console_output=True)
    logger = get_logger(__name__)

    # Create downloader
    downloader = ModelDownloader(args.models_dir, logger)

    if args.verify_only:
        # Verify installations
        logger.info("Verifying model installations...")
        status = downloader.verify_installations()

        for model_type, installed in status.items():
            status_str = "✓ Installed" if installed else "✗ Not installed"
            logger.info(f"{model_type}: {status_str}")

        # Exit with error if any required models missing
        if not all(status.values()):
            sys.exit(1)
    else:
        # Download models
        logger.info("Starting model download process...")

        if not args.skip_whisper:
            logger.info("Downloading Whisper models...")
            downloader.download_whisper_models(args.whisper_sizes)

        if not args.skip_sentence_transformers:
            logger.info("Downloading sentence transformer models...")
            downloader.download_sentence_transformers(args.sentence_transformers)

        if not args.skip_ollama:
            logger.info("Setting up Ollama models...")
            downloader.setup_ollama_models(args.ollama_models)

        # Create model registry
        downloader.create_model_registry()

        # Verify installations
        logger.info("\nVerifying installations...")
        status = downloader.verify_installations()

        all_good = True
        for model_type, installed in status.items():
            if installed:
                logger.info(f"✓ {model_type} models ready")
            else:
                logger.warning(f"✗ {model_type} models not found")
                all_good = False

        if all_good:
            logger.info("\n✓ All models successfully installed!")
        else:
            logger.warning("\n⚠ Some models are missing. Please check the logs above.")
            sys.exit(1)


if __name__ == "__main__":
    main()
