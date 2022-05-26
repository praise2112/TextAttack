"""
Attack Logs to CSV
========================
"""

import csv

import pandas as pd

from textattack.shared import AttackedText, logger

from .logger import Logger


class CSVLogger(Logger):
    """Logs attack results to a CSV."""

    def __init__(self, filename="results.csv", color_method="file"):
        logger.info(f"Logging to CSV at path {filename}")
        self.filename = filename
        self.color_method = color_method
        self.df = pd.DataFrame()
        self._flushed = True
        self.prev_result = None
        self.num_results = 1

    def log_attack_result(self, result):
        if self.prev_result is not None and self.prev_result.original_result != result.original_result:
            self.num_results += 1
        self.prev_result = result

        original_text, perturbed_text = result.diff_color(self.color_method)
        original_text = original_text.replace("\n", AttackedText.SPLIT_TOKEN)
        perturbed_text = perturbed_text.replace("\n", AttackedText.SPLIT_TOKEN)
        result_type = result.__class__.__name__.replace("AttackResult", "")
        row = {
            "result_index": self.num_results,
            "original_text": original_text,
            "perturbed_text": perturbed_text,
            "original_score": result.original_result.score,
            "perturbed_score": result.perturbed_result.score,
            "original_output": result.original_result.output,
            "perturbed_output": result.perturbed_result.output,
            "ground_truth_output": result.original_result.ground_truth_output,
            "num_queries": result.num_queries,
            "result_type": result_type,
        }
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self._flushed = False

    def flush(self):
        self.df.to_csv(self.filename, quoting=csv.QUOTE_NONNUMERIC, index=False)
        self._flushed = True

    def close(self):
        # self.fout.close()
        super().close()

    def __del__(self):
        if not self._flushed:
            logger.warning("CSVLogger exiting without calling flush().")
