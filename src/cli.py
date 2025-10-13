"""
Command Line Interface for Call Analytics System

Provides CLI commands for various operations like starting the UI,
rebuilding indexes, processing files, and system management.
"""

import argparse
import logging
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CallAnalyticsCLI:
    """Main CLI handler for the Call Analytics System"""

    def __init__(self):
        """Initialize CLI with argument parser"""
        self.parser = self.create_parser()

    def create_parser(self) -> argparse.ArgumentParser:
        """
        Create and configure argument parser.

        Returns:
            argparse.ArgumentParser: Configured parser
        """
        parser = argparse.ArgumentParser(
            prog="call-analytics",
            description="Call Analytics System CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # Add version argument
        parser.add_argument("--version", "-v", action="version", version="%(prog)s 1.0.0")

        # Add verbosity
        parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

        # Create subparsers for commands
        subparsers = parser.add_subparsers(
            title="Commands", dest="command", help="Available commands"
        )

        # UI command
        ui_parser = subparsers.add_parser("ui", help="Start the Streamlit UI")
        ui_parser.add_argument(
            "--port", type=int, default=8501, help="Port to run the UI on (default: 8501)"
        )
        ui_parser.add_argument(
            "--host", default="localhost", help="Host to bind to (default: localhost)"
        )

        # Process command
        process_parser = subparsers.add_parser("process", help="Process call data files")
        process_parser.add_argument("input", type=str, help="Input file or directory to process")
        process_parser.add_argument(
            "--output", type=str, help="Output directory for processed files"
        )
        process_parser.add_argument(
            "--type",
            choices=["csv", "audio", "auto"],
            default="auto",
            help="Type of processing (default: auto-detect)",
        )

        # Index command
        index_parser = subparsers.add_parser("index", help="Manage vector database index")
        index_parser.add_argument(
            "action", choices=["rebuild", "update", "clear"], help="Index action to perform"
        )
        index_parser.add_argument(
            "--collection",
            default="call_transcripts",
            help="Collection name (default: call_transcripts)",
        )

        # Models command
        models_parser = subparsers.add_parser("models", help="Manage ML models")
        models_parser.add_argument(
            "action", choices=["download", "list", "verify"], help="Model management action"
        )
        models_parser.add_argument(
            "--whisper-size",
            choices=["tiny", "base", "small", "medium", "large"],
            default="small",
            help="Whisper model size (default: small)",
        )

        # Setup command
        setup_parser = subparsers.add_parser("setup", help="Setup environment and dependencies")
        setup_parser.add_argument(
            "--skip-packages", action="store_true", help="Skip package installation"
        )
        setup_parser.add_argument("--skip-models", action="store_true", help="Skip model downloads")

        # Config command
        config_parser = subparsers.add_parser("config", help="Manage configuration")
        config_parser.add_argument(
            "action", choices=["show", "validate", "create"], help="Configuration action"
        )

        # Export command
        export_parser = subparsers.add_parser("export", help="Export data and analytics")
        export_parser.add_argument(
            "--format",
            choices=["csv", "excel", "json"],
            default="csv",
            help="Export format (default: csv)",
        )
        export_parser.add_argument("--output", type=str, required=True, help="Output file path")

        return parser

    def run_ui(self, args: argparse.Namespace) -> int:
        """
        Start the Streamlit UI.

        Args:
            args: Parsed command arguments

        Returns:
            int: Exit code
        """
        logger.info(f"Starting UI on {args.host}:{args.port}")

        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "src/ui/app.py",
            "--server.port",
            str(args.port),
            "--server.address",
            args.host,
        ]

        try:
            result = subprocess.run(cmd)
            return result.returncode
        except KeyboardInterrupt:
            logger.info("UI stopped by user")
            return 0
        except Exception as e:
            logger.error(f"Failed to start UI: {e}")
            return 1

    def process_files(self, args: argparse.Namespace) -> int:
        """
        Process call data files.

        Args:
            args: Parsed command arguments

        Returns:
            int: Exit code
        """
        input_path = Path(args.input)

        if not input_path.exists():
            logger.error(f"Input path does not exist: {input_path}")
            return 1

        try:
            # Lazy import to avoid loading heavy modules
            csv_suffixes = {".csv"}
            audio_suffixes = {".wav", ".mp3", ".m4a"}

            is_csv = args.type == "csv" or (
                args.type == "auto" and input_path.suffix.lower() in csv_suffixes
            )
            is_audio = args.type == "audio" or (
                args.type == "auto" and input_path.suffix.lower() in audio_suffixes
            )

            if is_csv:
                from src.core.csv_processor import CSVProcessor

                processor = CSVProcessor({"encoding": "utf-8"})
                logger.info(f"Processing CSV file: {input_path}")
                df = processor.read_csv(input_path)
                logger.info(f"Processed {len(df)} records")

            elif is_audio:
                from src.core.audio_processor import AudioProcessor

                processor = AudioProcessor(output_dir=Path(args.output or "data/processed"))
                logger.info(f"Processing audio file: {input_path}")
                result = processor.process_audio(input_path)
                logger.info(f"Audio processed: {result}")

            else:
                logger.error(f"Unsupported file type: {input_path.suffix}")
                return 1

            logger.info("Processing completed successfully")
            return 0

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return 1

    def manage_index(self, args: argparse.Namespace) -> int:
        """
        Manage vector database index.

        Args:
            args: Parsed command arguments

        Returns:
            int: Exit code
        """
        try:
            if args.action == "rebuild":
                logger.info(f"Rebuilding index for collection: {args.collection}")
                from scripts.rebuild_index import main as rebuild_main

                rebuild_main()

            elif args.action == "update":
                logger.info(f"Updating index for collection: {args.collection}")
                # Implementation for update

            elif args.action == "clear":
                logger.info(f"Clearing index for collection: {args.collection}")
                from src.vectordb.chroma_client import ChromaDBClient

                client = ChromaDBClient()
                client.reset_collection(args.collection)
                logger.info("Index cleared successfully")

            return 0

        except Exception as e:
            logger.error(f"Index operation failed: {e}")
            return 1

    def manage_models(self, args: argparse.Namespace) -> int:
        """
        Manage ML models.

        Args:
            args: Parsed command arguments

        Returns:
            int: Exit code
        """
        try:
            handlers: dict[str, Callable[[argparse.Namespace], int | None]] = {
                "download": self._download_models,
                "list": self._list_models,
                "verify": self._verify_models,
            }
            handler = handlers.get(args.action)
            if handler is None:
                logger.error("Unknown models action: %s", args.action)
                return 1

            result = handler(args)
            return 0 if result is None else result
        except Exception as e:
            logger.error(f"Model operation failed: {e}")
            return 1

    def _download_models(self, args: argparse.Namespace) -> None:
        logger.info(f"Downloading models (whisper size: {args.whisper_size})")
        from scripts.download_models import main as download_main

        download_main()

    def _list_models(self, _: argparse.Namespace) -> None:
        logger.info("Available models:")
        models_dir = Path("models")
        if not models_dir.exists():
            logger.info("  No models found")
            return

        for model_file in models_dir.rglob("*"):
            if not model_file.is_file():
                continue
            size_mb = model_file.stat().st_size / (1024 * 1024)
            rel_path = model_file.relative_to(models_dir)
            logger.info("  - %s (%.1f MB)", rel_path, size_mb)

    def _verify_models(self, _: argparse.Namespace) -> None:
        logger.info("Verifying models...")
        from src.ml import get_ml_capabilities

        capabilities = get_ml_capabilities()
        for key, value in capabilities.items():
            status = "✓" if value else "✗"
            logger.info(f"  {status} {key}: {value}")

    def setup_environment(self) -> int:
        """
        Setup environment and dependencies.

        Returns:
            int: Exit code
        """
        try:
            logger.info("Setting up environment...")
            from scripts.setup_environment import main as setup_main

            setup_main()
            return 0
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return 1

    def manage_config(self, args: argparse.Namespace) -> int:
        """
        Manage configuration.

        Args:
            args: Parsed command arguments

        Returns:
            int: Exit code
        """
        try:
            handlers: dict[str, Callable[[argparse.Namespace], int | None]] = {
                "show": self._show_config,
                "validate": self._validate_config,
                "create": self._create_config,
            }
            handler = handlers.get(args.action)
            if handler is None:
                logger.error("Unknown config action: %s", args.action)
                return 1

            result = handler(args)
            return 0 if result is None else result
        except Exception as e:
            logger.error(f"Config operation failed: {e}")
            return 1

    def _show_config(self, _: argparse.Namespace) -> None:
        import toml

        config_dir = Path("config")
        for config_file in config_dir.glob("*.toml"):
            logger.info(f"\n=== {config_file.name} ===")
            with open(config_file) as file:
                config = toml.load(file)
            for key, value in config.items():
                logger.info(f"{key}: {value}")

    def _validate_config(self, _: argparse.Namespace) -> int:
        import toml

        logger.info("Validating configuration files...")
        all_valid = True
        for config_file in Path("config").glob("*.toml"):
            try:
                with open(config_file) as file:
                    toml.load(file)
                logger.info(f"  ✓ {config_file.name} is valid")
            except Exception as e:
                logger.error(f"  ✗ {config_file.name} is invalid: {e}")
                all_valid = False
        return 0 if all_valid else 1

    def _create_config(self, _: argparse.Namespace) -> None:
        logger.info("Creating default configuration files...")
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        # Create default configs
        # Implementation for creating default configs

    def export_data(self, args: argparse.Namespace) -> int:
        """
        Export data and analytics.

        Args:
            args: Parsed command arguments

        Returns:
            int: Exit code
        """
        try:
            logger.info(f"Exporting data to {args.output} in {args.format} format")

            from src.core.csv_processor import CSVExporter
            from src.core.storage_manager import StorageManager

            storage = StorageManager(base_path=Path("data"))
            data = storage.load_all_records()

            if args.format == "csv":
                exporter = CSVExporter()
                exporter.export_to_csv(data, Path(args.output))
            elif args.format == "excel":
                exporter = CSVExporter()
                exporter.export_to_excel(data, Path(args.output))
            elif args.format == "json":
                import json

                with open(args.output, "w") as f:
                    json.dump(data, f, indent=2, default=str)

            logger.info(f"Export completed: {args.output}")
            return 0

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return 1

    def run(self, argv: list[str] | None = None) -> int:
        """
        Main CLI entry point.

        Args:
            argv: Command line arguments (defaults to sys.argv)

        Returns:
            int: Exit code
        """
        args = self.parser.parse_args(argv)

        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Route to appropriate handler
        if args.command == "ui":
            return self.run_ui(args)
        elif args.command == "process":
            return self.process_files(args)
        elif args.command == "index":
            return self.manage_index(args)
        elif args.command == "models":
            return self.manage_models(args)
        elif args.command == "setup":
            return self.setup_environment()
        elif args.command == "config":
            return self.manage_config(args)
        elif args.command == "export":
            return self.export_data(args)
        else:
            self.parser.print_help()
            return 0


def main(argv: list[str] | None = None) -> int:
    """
    Main CLI entry point.

    Args:
        argv: Command line arguments

    Returns:
        int: Exit code
    """
    try:
        cli = CallAnalyticsCLI()
        return cli.run(argv)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
