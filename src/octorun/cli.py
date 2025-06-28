"""Command-line interface for octorun."""

import argparse
import sys
from typing import List, Optional

from . import __version__ as version


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="octorun",
        description="A command-line tool for octorun",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {}".format(version),
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND",
    )
    
    # Run command (example for running something)
    run_parser = subparsers.add_parser(
        "run",
        help="Run a task or script",
    )
    run_parser.add_argument(
        "script",
        help="Script or task to run",
    )

    
    
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    """Handle the run command."""
    if args.verbose:
        print(f"Running script: {args.script}")

    # Here you would implement the actual run logic
    print(f"Running {args.script}...")
    print("(This is where your run logic would go)")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if args.verbose:
        print(f"octorun v0.1.0")
        print(f"Command: {args.command}")
    
    # Dispatch to command handler
    if args.command == "run":
        return cmd_run(args)
    else:
        parser.print_help()
        return 1


def cli_main() -> None:
    """Entry point for console script."""
    sys.exit(main())


if __name__ == "__main__":
    cli_main()
