"""Command-line interface for AAM training and inference."""

from aam.cli.utils import (
    setup_logging,
    setup_device,
    setup_expandable_segments,
    setup_random_seed,
    validate_file_path,
    validate_arguments,
)

import click

from aam.cli.predict import predict
from aam.cli.pretrain import pretrain
from aam.cli.rf_baseline import rf_baseline
from aam.cli.train import train


@click.group()
def cli():
    """AAM (Attention All Microbes) - Deep learning for microbial sequencing data."""
    pass


# Register commands
cli.add_command(predict)
cli.add_command(pretrain)
cli.add_command(rf_baseline)
cli.add_command(train)


def main():
    """Main entry point for CLI."""
    cli()


__all__ = [
    "cli",
    "main",
    "setup_logging",
    "setup_device",
    "setup_expandable_segments",
    "setup_random_seed",
    "validate_file_path",
    "validate_arguments",
]
