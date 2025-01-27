"""
Attack Logs to WandB
========================
"""


from textattack.shared.utils import LazyLoader, html_table_from_rows

from .logger import Logger


class WeightsAndBiasesLogger(Logger):
    """Logs attack results to Weights & Biases."""

    def __init__(self, **kwargs):

        global wandb
        wandb = LazyLoader("wandb", globals(), "wandb")

        wandb.init(**kwargs)
        self.kwargs = kwargs
        self.project_name = wandb.run.project_name()
        self._result_table_rows = []
        self.prev_result = None
        self.num_results = 1

    def __setstate__(self, state):
        global wandb
        wandb = LazyLoader("wandb", globals(), "wandb")

        self.__dict__ = state
        wandb.init(resume=True, **self.kwargs)

    def log_summary_rows(self, rows, title, window_id):
        table = wandb.Table(columns=["Attack Results", ""])
        for row in rows:
            if isinstance(row[1], str):
                try:
                    row[1] = row[1].replace("%", "")
                    row[1] = float(row[1])
                except ValueError as e:
                    raise ValueError(
                        f'Unable to convert row value "{row[1]}" for Attack Result "{row[0]}" into float') from e
            table.add_data(*row)
            metric_name, metric_score = row
            wandb.run.summary[metric_name] = metric_score
        wandb.log({"attack_params": table})

    def _log_result_table(self):
        """Weights & Biases doesn't have a feature to automatically aggregate
        results across timesteps and display the full table.

        Therefore, we have to do it manually.
        """
        result_table = html_table_from_rows(
            self._result_table_rows, header=["", "Original Input", "Perturbed Input"]
        )
        wandb.log({"results": wandb.Html(result_table)})

    def log_attack_result(self, result):
        if self.prev_result is not None and self.prev_result.original_result != result.original_result:
            self.num_results += 1
        self.prev_result = result

        try:
            original_text_colored, perturbed_text_colored = result.diff_color(
                color_method="html"
            )
        except IndexError:
            original_text_colored, perturbed_text_colored = result.diff_color(
                color_method=None
            )
        result_num = len(self._result_table_rows)
        self._result_table_rows.append(
            [
                f"<b>Result {result_num}</b>",
                original_text_colored,
                perturbed_text_colored,
            ]
        )
        result_diff_table = html_table_from_rows(
            [[original_text_colored, perturbed_text_colored]]
        )
        result_diff_table = wandb.Html(result_diff_table)
        result_type = result.__class__.__name__.replace("AttackResult", "")
        wandb.log(
            {
                "result_index": self.num_results,
                "result": result_diff_table,
                "original_output": result.original_result.output,
                "perturbed_output": result.perturbed_result.output,
                "original_score": result.original_result.score,
                "perturbed_score": result.perturbed_result.score,
                "ground_truth_output": result.original_result.ground_truth_output,
                "num_queries": result.num_queries,
                "result_type": result_type,
            }
        )
        self._log_result_table()

    def log_sep(self):
        self.fout.write("-" * 90 + "\n")
