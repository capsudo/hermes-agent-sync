"""Standalone development entry point; no profile access on import or invocation."""

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments without reading or changing profile files."""
    parser = argparse.ArgumentParser(
        prog="hermes-agent-sync",
        description="Development CLI for Hermes Profile Sync.",
        epilog="Synchronization is not implemented yet.",
    )
    parser.parse_args(argv)
    parser.print_help()
    return 0
