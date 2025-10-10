"""
Environment Setup Script for Call Analytics System

This script sets up the complete environment including directories,
configuration files, dependencies, and initial data structures.
"""

import argparse
import importlib.util
import logging
import subprocess
import sys
from pathlib import Path

import toml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class EnvironmentSetup:
    """
    Handles complete environment setup for the Call Analytics System.
    """

    # Required Python version
    REQUIRED_PYTHON = (3, 13)

    # Required directories
    REQUIRED_DIRS = [
        'data',
        'data/raw',
        'data/processed',
        'data/vector_db',
        'data/exports',
        'models',
        'models/whisper',
        'models/sentence_transformers',
        'logs',
        'backups',
        'config',
        'temp'
    ]

    # Required Python packages
    REQUIRED_PACKAGES = [
        'streamlit>=1.28.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'plotly>=5.17.0',
        'pydantic>=2.0.0',
        'toml>=0.10.2',
        'python-dateutil>=2.8.2',
        'pytz>=2023.3',
        'phonenumbers>=8.13.0',
        'tqdm>=4.65.0',
        'requests>=2.31.0',
        'psutil>=5.9.0',
        'chardet>=5.2.0',
        'xlsxwriter>=3.1.0',
        'openpyxl>=3.1.0',
        'sentence-transformers>=2.2.0',
        'chromadb>=0.4.0',
        'faster-whisper>=0.9.0',
        'soundfile>=0.12.0',
        'librosa>=0.10.0',
        'pytest>=7.4.0',
        'black>=23.0.0',
        'pylint>=3.0.0'
    ]

    # Optional packages
    OPTIONAL_PACKAGES = [
        'ollama>=0.1.0',
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'accelerate>=0.20.0'
    ]

    def __init__(self, base_dir: Path, logger: logging.Logger):
        """
        Initialize environment setup.

        Args:
            base_dir: Base directory for the project
            logger: Logger instance
        """
        self.base_dir = base_dir
        self.logger = logger

    def check_python_version(self) -> bool:
        """
        Check if Python version meets requirements.

        Returns:
            True if version is adequate, False otherwise
        """
        current_version = sys.version_info[:2]
        required = self.REQUIRED_PYTHON

        self.logger.info(f"Python version: {sys.version}")

        if current_version < required:
            self.logger.error(
                f"Python {required[0]}.{required[1]} or higher is required. "
                f"Current version: {current_version[0]}.{current_version[1]}"
            )
            return False

        self.logger.info("✓ Python version check passed")
        return True

    def create_directories(self) -> bool:
        """
        Create required directory structure.

        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Creating directory structure...")

        try:
            for dir_path in self.REQUIRED_DIRS:
                full_path = self.base_dir / dir_path
                full_path.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created directory: {full_path}")

            self.logger.info("✓ Directory structure created")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create directories: {e}")
            return False

    def install_packages(self, upgrade: bool = False) -> bool:
        """
        Install required Python packages.

        Args:
            upgrade: Whether to upgrade existing packages

        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Installing required packages...")

        # Prepare pip command
        pip_cmd = [sys.executable, '-m', 'pip', 'install']
        if upgrade:
            pip_cmd.append('--upgrade')

        # Install required packages
        for package in self.REQUIRED_PACKAGES:
            try:
                self.logger.info(f"Installing {package}...")
                result = subprocess.run(
                    pip_cmd + [package],
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    self.logger.error(f"Failed to install {package}: {result.stderr}")
                    return False

            except Exception as e:
                self.logger.error(f"Error installing {package}: {e}")
                return False

        # Try to install optional packages
        self.logger.info("Installing optional packages...")
        for package in self.OPTIONAL_PACKAGES:
            try:
                result = subprocess.run(
                    pip_cmd + [package],
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:
                    self.logger.info(f"✓ Installed optional package: {package}")
                else:
                    self.logger.warning(f"Could not install optional package: {package}")

            except Exception as e:
                self.logger.warning(f"Skipping optional package {package}: {e}")

        self.logger.info("✓ Package installation complete")
        return True

    def create_config_files(self) -> bool:
        """
        Create default configuration files.

        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Creating configuration files...")

        try:
            # Create app.toml
            app_config = {
                'app': {
                    'name': 'Call Analytics System',
                    'version': '1.0.0',
                    'debug': False,
                    'theme': 'dark'
                },
                'paths': {
                    'data': 'data',
                    'models': 'models',
                    'logs': 'logs',
                    'exports': 'data/exports',
                    'temp': 'temp'
                },
                'limits': {
                    'max_file_size_mb': 100,
                    'max_batch_size': 1000,
                    'max_concurrent_jobs': 5
                }
            }

            app_config_path = self.base_dir / 'config' / 'app.toml'
            with open(app_config_path, 'w') as f:
                toml.dump(app_config, f)
            self.logger.info(f"Created {app_config_path}")

            # Create models.toml
            models_config = {
                'whisper': {
                    'enabled': True,
                    'model_size': 'small',
                    'device': 'cpu',
                    'compute_type': 'int8',
                    'language': 'en',
                    'initial_prompt': None
                },
                'embeddings': {
                    'provider': 'sentence_transformers',
                    'model': 'all-MiniLM-L6-v2',
                    'dimension': 384,
                    'batch_size': 32
                },
                'llm': {
                    'provider': 'ollama',
                    'model': 'llama3:8b',
                    'temperature': 0.7,
                    'max_tokens': 2000
                }
            }

            models_config_path = self.base_dir / 'config' / 'models.toml'
            with open(models_config_path, 'w') as f:
                toml.dump(models_config, f)
            self.logger.info(f"Created {models_config_path}")

            # Create .env.example
            env_example = """# Environment variables for Call Analytics System

# API Keys (if using cloud services)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Database
DATABASE_URL=sqlite:///data/analytics.db
VECTOR_DB_PATH=data/vector_db

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost

# Security
SECRET_KEY=your_secret_key_here
"""

            env_path = self.base_dir / '.env.example'
            with open(env_path, 'w') as f:
                f.write(env_example)
            self.logger.info(f"Created {env_path}")

            self.logger.info("✓ Configuration files created")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create config files: {e}")
            return False

    def setup_streamlit_config(self) -> bool:
        """
        Create Streamlit configuration.

        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Setting up Streamlit configuration...")

        try:
            # Create .streamlit directory
            streamlit_dir = self.base_dir / '.streamlit'
            streamlit_dir.mkdir(exist_ok=True)

            # Create config.toml for Streamlit
            streamlit_config = {
                'theme': {
                    'primaryColor': '#1f77b4',
                    'backgroundColor': '#0e1117',
                    'secondaryBackgroundColor': '#262730',
                    'textColor': '#fafafa',
                    'font': 'sans serif'
                },
                'server': {
                    'port': 8501,
                    'address': 'localhost',
                    'headless': True,
                    'runOnSave': True,
                    'maxUploadSize': 100
                },
                'browser': {
                    'gatherUsageStats': False
                }
            }

            config_path = streamlit_dir / 'config.toml'
            with open(config_path, 'w') as f:
                toml.dump(streamlit_config, f)

            self.logger.info("✓ Streamlit configuration created")
            return True

        except Exception as e:
            self.logger.error(f"Failed to setup Streamlit config: {e}")
            return False

    def create_sample_data(self) -> bool:
        """
        Create sample assets (tabular and audio) for testing.

        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Creating sample data...")

        csv_success = self._create_sample_csv_data()
        voice_success = self._create_sample_voice_data()

        if not csv_success:
            self.logger.warning("Sample CSV generation failed; see logs above")

        if not voice_success:
            self.logger.warning("Sample voice generation failed; see logs above")

        return csv_success and voice_success

    def _create_sample_csv_data(self) -> bool:
        """Create CSV sample data matching the call import format."""
        try:
            import random
            from datetime import datetime, timedelta

            import pandas as pd

            num_records = 100

            phone_numbers = [f"+1{random.randint(2000000000, 9999999999)}" for _ in range(50)]
            agents = [f"agent_{i:03d}" for i in range(1, 11)]
            campaigns = ['sales', 'support', 'billing', 'retention', 'survey']
            outcomes = ['connected', 'no_answer', 'voicemail', 'busy', 'failed']
            call_types = ['inbound', 'outbound']

            records = []
            base_date = datetime.now() - timedelta(days=30)

            for i in range(num_records):
                record = {
                    'call_id': f"CALL_{i:06d}",
                    'phone_number': random.choice(phone_numbers),
                    'timestamp': base_date + timedelta(
                        days=random.randint(0, 30),
                        hours=random.randint(8, 18),
                        minutes=random.randint(0, 59)
                    ),
                    'duration': random.randint(30, 600),
                    'agent_id': random.choice(agents),
                    'campaign': random.choice(campaigns),
                    'outcome': random.choice(outcomes),
                    'call_type': random.choice(call_types),
                    'revenue': random.choice([0, 0, 0, random.uniform(10, 500)]),
                    'notes': random.choice([
                        '',
                        'Customer satisfied',
                        'Follow-up required',
                        'Technical issue resolved',
                        'Billing inquiry',
                        'Product complaint'
                    ])
                }
                records.append(record)

            df = pd.DataFrame(records)
            sample_file = self.base_dir / 'data' / 'raw' / 'sample_calls.csv'
            df.to_csv(sample_file, index=False)

            self.logger.info(f"✓ Created sample CSV data with {num_records} records")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create sample CSV data: {e}")
            return False

    def _create_sample_voice_data(self) -> bool:
        """Create sample audio files with transcripts for voice ingestion demos."""
        try:
            import random
            from datetime import datetime, timedelta

            import pandas as pd

            audio_dir = self.base_dir / 'data' / 'raw' / 'sample_audio'
            transcripts_dir = audio_dir / 'transcripts'
            audio_dir.mkdir(parents=True, exist_ok=True)
            transcripts_dir.mkdir(parents=True, exist_ok=True)

            base_date = datetime.now() - timedelta(days=7)
            phone_numbers = [f"+1{random.randint(2000000000, 9999999999)}" for _ in range(20)]
            agents = [
                {'id': 'agent_101', 'name': 'Alex Rivera'},
                {'id': 'agent_204', 'name': 'Priya Malhotra'},
                {'id': 'agent_318', 'name': 'Taylor Chen'},
                {'id': 'agent_422', 'name': 'Morgan Blake'},
                {'id': 'agent_537', 'name': 'Jamie Patel'}
            ]

            sample_scripts = [
                {
                    'call_id': 'CALL_AUDIO_001',
                    'script': (
                        "Hello, thanks for calling Acme Support. This is Alex speaking. "
                        "I understand you're seeing an unexpected billing charge. "
                        "I'd be happy to take a look at that for you."
                    ),
                    'notes': 'Resolved billing inquiry on the call.',
                    'call_topic': 'billing_correction',
                    'campaign': 'billing_support',
                    'outcome': 'resolved',
                    'call_type': 'inbound',
                    'revenue': 0.0,
                    'tags': ['billing', 'invoice', 'refund'],
                    'sentiment': 'concerned'
                },
                {
                    'call_id': 'CALL_AUDIO_002',
                    'script': (
                        "Good afternoon, you've reached the Acme sales desk. This is Priya. "
                        "I'm calling to follow up on the demo you attended last week "
                        "and see if you had any questions."
                    ),
                    'notes': 'Left voicemail requesting callback.',
                    'call_topic': 'sales_follow_up',
                    'campaign': 'midmarket_outreach',
                    'outcome': 'voicemail',
                    'call_type': 'outbound',
                    'revenue': 0.0,
                    'tags': ['sales', 'follow_up'],
                    'sentiment': 'neutral'
                },
                {
                    'call_id': 'CALL_AUDIO_003',
                    'script': (
                        "Hi Jamie, it's Taylor from Acme Customer Success. "
                        "I'm checking in to confirm that the latest firmware update "
                        "resolved the disconnect issue you reported."
                    ),
                    'notes': 'Customer confirmed issue resolved; scheduled follow-up email.',
                    'call_topic': 'technical_support',
                    'campaign': 'customer_success',
                    'outcome': 'resolved',
                    'call_type': 'outbound',
                    'revenue': random.choice([0.0, 149.0, 299.0]),
                    'tags': ['support', 'firmware'],
                    'sentiment': 'positive'
                },
                {
                    'call_id': 'CALL_AUDIO_004',
                    'script': (
                        "Hi, this is Morgan from Acme Renewals. "
                        "I'm reaching out to discuss your upcoming subscription renewal "
                        "and share the loyalty discount that's available this quarter."
                    ),
                    'notes': 'Customer agreed to renew with discount applied.',
                    'call_topic': 'renewal_negotiation',
                    'campaign': 'retention_push',
                    'outcome': 'upsold',
                    'call_type': 'outbound',
                    'revenue': 499.0,
                    'tags': ['renewal', 'discount'],
                    'sentiment': 'optimistic'
                },
                {
                    'call_id': 'CALL_AUDIO_005',
                    'script': (
                        "Hello, you've reached Acme Surveys. "
                        "We're collecting quick feedback about your recent "
                        "installation appointment. "
                        "Do you have two minutes to answer three questions?"
                    ),
                    'notes': (
                        'Captured NPS response and forwarded to analytics team.'
                    ),
                    'call_topic': 'customer_feedback',
                    'campaign': 'nps_automation',
                    'outcome': 'completed',
                    'call_type': 'outbound',
                    'revenue': 0.0,
                    'tags': ['survey', 'feedback'],
                    'sentiment': 'neutral'
                }
            ]

            metadata_rows = []
            method_stats = {'pyttsx3': 0, 'say': 0, 'tone_fallback': 0}

            for idx, sample in enumerate(sample_scripts, start=1):
                audio_name = f"sample_call_{idx:02d}.wav"
                audio_path = audio_dir / audio_name
                transcript_path = transcripts_dir / f"{audio_path.stem}.txt"

                generated, method_used = self._generate_voice_sample(
                    sample['script'],
                    audio_path
                )

                if method_used in method_stats:
                    method_stats[method_used] += 1
                else:
                    method_stats[method_used] = 1

                transcript_path.write_text(sample['script'], encoding='utf-8')

                # Compute duration if possible
                duration_seconds = self._measure_audio_duration(audio_path)

                agent = random.choice(agents)
                record = {
                    'call_id': sample['call_id'],
                    'audio_file': audio_name,
                    'transcript_file': str(transcript_path.relative_to(audio_dir)),
                    'transcript': sample['script'],
                    'phone_number': random.choice(phone_numbers),
                    'timestamp': base_date + timedelta(
                        days=random.randint(0, 7),
                        hours=random.randint(8, 18),
                        minutes=random.randint(0, 59)
                    ),
                    'duration_seconds': duration_seconds,
                    'agent_id': agent['id'],
                    'agent_name': agent['name'],
                    'campaign': sample['campaign'],
                    'outcome': sample['outcome'],
                    'call_type': sample['call_type'],
                    'call_topic': sample['call_topic'],
                    'sentiment': sample['sentiment'],
                    'tags': ",".join(sample['tags']),
                    'revenue': sample['revenue'],
                    'notes': sample['notes']
                }

                metadata_rows.append(record)

                if not generated:
                    self.logger.warning(
                        "Generated synthetic tone for %s; install pyttsx3 or use "
                        "macOS 'say' for real speech",
                        audio_name,
                    )

            metadata_df = pd.DataFrame(metadata_rows)
            metadata_file = audio_dir / 'sample_audio_metadata.csv'
            metadata_df.to_csv(metadata_file, index=False)

            created_files = len(metadata_rows)
            if created_files:
                details = ", ".join(
                    f"{method}:{count}" for method, count in method_stats.items() if count
                )
                self.logger.info(
                    "✓ Created %s sample audio files in %s (%s)",
                    created_files,
                    audio_dir,
                    details or 'no audio generated'
                )

            return True

        except Exception as e:
            self.logger.error(f"Failed to create sample voice data: {e}")
            return False

    def _generate_voice_sample(self, text: str, output_path: Path) -> tuple[bool, str]:
        """Create a spoken WAV file for the provided text.

        Returns a tuple of (success flag, method label).
        """
        try:
            import pyttsx3  # type: ignore

            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.save_to_file(text, str(output_path))
            engine.runAndWait()
            return True, 'pyttsx3'

        except Exception as pyttsx_error:
            self.logger.debug(f"pyttsx3 not available or failed: {pyttsx_error}")

        try:
            import shutil
            say_path = shutil.which('say')
            if say_path:
                subprocess.run(
                    [
                        say_path,
                        '-o', str(output_path),
                        '--file-format=WAVE',
                        '--data-format=LEF32@16000',
                        text
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                return True, 'say'
        except Exception as say_error:
            self.logger.debug(f"macOS say command failed: {say_error}")

        try:
            self._write_tone_sample(output_path)
            return False, 'tone_fallback'
        except Exception as tone_error:
            self.logger.error(f"Failed to write fallback tone sample: {tone_error}")
            raise

    def _write_tone_sample(self, output_path: Path, duration: float = 3.0) -> None:
        """Write a simple sine wave tone when no TTS engine is available."""
        import math
        import struct
        import wave

        sample_rate = 16000
        frequency = 440
        amplitude = 16000
        frame_count = int(duration * sample_rate)

        with wave.open(str(output_path), 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)

            for frame_index in range(frame_count):
                angle = 2 * math.pi * frequency * (frame_index / sample_rate)
                value = int(amplitude * math.sin(angle))
                wav_file.writeframes(struct.pack('<h', value))

    @staticmethod
    def _measure_audio_duration(audio_path: Path) -> float:
        """Read actual WAV duration in seconds, returning 0 on failure."""
        try:
            import contextlib
            import wave

            import aifc

            try:
                import soundfile as sf  # type: ignore

                info = sf.info(str(audio_path))
                if info.frames and info.samplerate:
                    return round(info.frames / float(info.samplerate), 2)
                if getattr(info, 'duration', 0):
                    return round(info.duration, 2)
            except (ImportError, RuntimeError):
                # soundfile not available or unsupported format; fall back to stdlib readers
                pass

            for opener in (wave.open, aifc.open):
                try:
                    with contextlib.closing(opener(str(audio_path), 'rb')) as af:
                        frames = af.getnframes()
                        framerate = af.getframerate()
                        if framerate:
                            return round(frames / float(framerate), 2)
                except (wave.Error, aifc.Error, FileNotFoundError):
                    continue
        except Exception as exc:
            logger = logging.getLogger(__name__)
            logger.debug("Unable to measure duration for %s: %s", audio_path, exc)
        return 0.0

    def verify_installation(self) -> tuple[bool, dict[str, bool]]:
        """
        Verify the installation is complete and functional.

        Returns:
            Tuple of (overall success, detailed status dict)
        """
        self.logger.info("Verifying installation...")

        status = {}

        # Check directories
        status['directories'] = all(
            (self.base_dir / dir_path).exists()
            for dir_path in self.REQUIRED_DIRS
        )

        # Check config files
        status['config_files'] = all([
            (self.base_dir / 'config' / 'app.toml').exists(),
            (self.base_dir / 'config' / 'models.toml').exists()
        ])

        # Check Python packages via module availability
        core_modules = ['pandas', 'plotly', 'streamlit']
        status['core_packages'] = all(
            importlib.util.find_spec(module) is not None
            for module in core_modules
        )

        optional_modules = {
            'pytorch': 'torch',
            'ollama': 'ollama',
        }
        for status_key, module_name in optional_modules.items():
            status[status_key] = importlib.util.find_spec(module_name) is not None

        # Overall status
        required_checks = ['directories', 'config_files', 'core_packages']
        overall = all(status.get(check, False) for check in required_checks)

        return overall, status

    def print_next_steps(self) -> None:
        """
        Print next steps for the user after setup.
        """
        print("\n" + "="*60)
        print("✓ Environment setup complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. Download models:")
        print("   python scripts/download_models.py")
        print("\n2. Start the application:")
        print("   streamlit run src/ui/app.py")
        print("\n3. Access the application:")
        print("   http://localhost:8501")
        print("\nOptional:")
        print("- Configure Ollama for LLM support")
        print("- Customize config files in config/")
        print("- Upload your call data via the web interface")
        print("="*60)


def setup_logging_simple() -> logging.Logger:
    """
    Simple logging setup for the setup script.

    Returns:
        Logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def main():
    """
    Main function to run the environment setup.
    """
    parser = argparse.ArgumentParser(
        description='Setup environment for Call Analytics System'
    )

    parser.add_argument(
        '--base-dir',
        type=Path,
        default=Path.cwd(),
        help='Base directory for the project'
    )

    parser.add_argument(
        '--skip-packages',
        action='store_true',
        help='Skip Python package installation'
    )

    parser.add_argument(
        '--skip-sample-data',
        action='store_true',
        help='Skip creating sample data'
    )

    parser.add_argument(
        '--upgrade-packages',
        action='store_true',
        help='Upgrade existing packages'
    )

    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing installation'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging_simple()

    # Create setup instance
    setup = EnvironmentSetup(args.base_dir, logger)

    if args.verify_only:
        # Verification only
        overall, status = setup.verify_installation()

        logger.info("\nInstallation status:")
        for component, installed in status.items():
            status_icon = "✓" if installed else "✗"
            logger.info(f"  {status_icon} {component}")

        if overall:
            logger.info("\n✓ Installation verified successfully!")
            sys.exit(0)
        else:
            logger.error("\n✗ Installation incomplete")
            sys.exit(1)

    # Run full setup
    logger.info("Starting Call Analytics System environment setup...")
    logger.info(f"Base directory: {args.base_dir}")

    # Check Python version
    if not setup.check_python_version():
        sys.exit(1)

    # Create directories
    if not setup.create_directories():
        sys.exit(1)

    # Install packages
    if (
        not args.skip_packages
        and not setup.install_packages(upgrade=args.upgrade_packages)
    ):
        logger.error("Package installation failed")
        sys.exit(1)

    # Create config files
    if not setup.create_config_files():
        sys.exit(1)

    # Setup Streamlit
    if not setup.setup_streamlit_config():
        sys.exit(1)

    # Create sample data
    if not args.skip_sample_data:
        setup.create_sample_data()

    # Verify installation
    overall, status = setup.verify_installation()

    logger.info("\nInstallation summary:")
    for component, installed in status.items():
        status_icon = "✓" if installed else "✗"
        logger.info(f"  {status_icon} {component}")

    if overall:
        setup.print_next_steps()
    else:
        logger.warning("\nSetup completed with some optional components missing.")
        logger.info("The core system should work, but some features may be limited.")
        setup.print_next_steps()


if __name__ == '__main__':
    main()
