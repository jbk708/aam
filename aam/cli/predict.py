"""Predict command for AAM CLI."""

import click
import torch
import logging
import pandas as pd
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader

from aam.data.biom_loader import BIOMLoader
from aam.data.dataset import ASVDataset, collate_fn
from aam.models.sequence_predictor import SequencePredictor
from aam.cli.utils import (
    setup_device,
    validate_file_path,
    validate_arguments,
)


@click.command()
@click.option("--model", required=True, type=click.Path(exists=True), help="Path to trained model checkpoint")
@click.option("--table", required=True, type=click.Path(exists=True), help="Path to BIOM table file")
@click.option("--output", required=True, type=click.Path(), help="Output file for predictions")
@click.option("--batch-size", default=8, type=int, help="Batch size for inference")
@click.option("--device", default="cuda", type=click.Choice(["cuda", "cpu"]), help="Device to use")
@click.option(
    "--no-sequence-cache",
    is_flag=True,
    help="Disable sequence tokenization cache (enabled by default for faster inference)",
)
def predict(
    model: str,
    table: str,
    output: str,
    batch_size: int,
    device: str,
    no_sequence_cache: bool,
):
    """Run inference with trained AAM model."""
    try:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logger = logging.getLogger(__name__)
        logger.info("Starting AAM inference")

        validate_file_path(model, "Model checkpoint")
        validate_file_path(table, "BIOM table")

        validate_arguments(batch_size=batch_size)

        device_obj = setup_device(device)

        logger.info("Loading model...")
        checkpoint = torch.load(model, map_location=device_obj, weights_only=True)

        if "model_state_dict" in checkpoint:
            model_state = checkpoint["model_state_dict"]
            model_config = checkpoint.get("config", {})
        else:
            model_state = checkpoint
            model_config = {}

        logger.info("Loading data...")
        biom_loader = BIOMLoader()
        table_obj = biom_loader.load_table(table)

        dataset = ASVDataset(
            table=table_obj,
            max_bp=model_config.get("max_bp", 150),
            token_limit=model_config.get("token_limit", 1024),
            cache_sequences=not no_sequence_cache,
        )

        inference_collate = partial(
            collate_fn,
            token_limit=model_config.get("token_limit", 1024),
            unifrac_distances=None,
            unifrac_metric="unweighted",
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=inference_collate,
            pin_memory=device == "cuda",
        )

        logger.info("Creating model...")
        model_obj = SequencePredictor(
            encoder_type=model_config.get("encoder_type", "unifrac"),
            vocab_size=6,
            embedding_dim=model_config.get("embedding_dim", 128),
            max_bp=model_config.get("max_bp", 150),
            token_limit=model_config.get("token_limit", 1024),
            out_dim=model_config.get("out_dim", 1),
            is_classifier=model_config.get("is_classifier", False),
        )
        model_obj.load_state_dict(model_state)
        model_obj.to(device_obj)
        model_obj.eval()

        logger.info("Running inference...")
        predictions = []
        sample_ids_list = []

        with torch.no_grad():
            for batch in dataloader:
                tokens = batch["tokens"].to(device_obj)
                outputs = model_obj(tokens, return_nucleotides=False)

                if "target_prediction" in outputs:
                    pred = outputs["target_prediction"].cpu().numpy()
                    if pred.ndim == 1:
                        predictions.extend(pred.tolist())
                    elif pred.ndim == 2 and pred.shape[1] == 1:
                        predictions.extend(pred.squeeze(1).tolist())
                    else:
                        predictions.extend([p.tolist() for p in pred])
                    sample_ids_list.extend(batch["sample_ids"])

        logger.info(f"Writing predictions to {output}...")
        if predictions and isinstance(predictions[0], list):
            pred_cols = {f"prediction_{i}": [p[i] for p in predictions] for i in range(len(predictions[0]))}
            output_df = pd.DataFrame({"sample_id": sample_ids_list, **pred_cols})
        else:
            output_df = pd.DataFrame({"sample_id": sample_ids_list, "prediction": predictions})
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, sep="\t", index=False)

        logger.info(f"Inference completed. Predictions saved to {output}")

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise click.ClickException(str(e))
