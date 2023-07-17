import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import click
from pathlib import Path
import logging

from sklearn.metrics import ConfusionMatrixDisplay

log = logging.getLogger(__name__)


@click.command()
@click.argument("source", default="predictions.csv")
def main(source):
    from rich.logging import RichHandler

    logging.basicConfig(
        level="DEBUG",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    log.setLevel("DEBUG")
    eval_metrics(source)


def eval_metrics(source: Path):
    source = Path(source)
    df = pd.read_csv(source)
    df.fillna("none", inplace=True)

    # collapsing s1 predictions
    df.loc[df["pred_task"] == "s1_left", "pred_task"] = "s1"
    df.loc[df["pred_task"] == "s1_right", "pred_task"] = "s1"

    # names in csv
    level = ["task", "activity", "acquisition", "frame"]

    gt_corridor = np.unique(df["task"])
    metrics_df = pd.DataFrame(gt_corridor, columns=["gt_corridor"])

    total = dict()
    for t in gt_corridor:
        total.update({t: len(df[df["task"] == t])})
    metrics_df["total"] = metrics_df["gt_corridor"].map(total)

    add_correct_frames(df, metrics_df, gt_corridor, level, show_correct=False)

    # Overall things
    total_row = dict()
    for lev in level:
        correct = np.sum(df[lev] == df[f"pred_{lev}"])
        acc = correct / len(df)
        log.info(f"{lev} accuracy: {acc:.2f}")
        total_row.update({f"{lev}_acc": acc})

    row = pd.DataFrame(
        dict(
            gt_corridor=["total"],
            total=[len(df)],
            **dict((k, [v]) for k, v in total_row.items()),
        ),
    )
    metrics_df = pd.concat([metrics_df, row], ignore_index=True)

    # Confusion matrix
    for lev in level:
        disp = ConfusionMatrixDisplay.from_predictions(
            df[lev],
            df[f"pred_{lev}"],
            normalize="true",
        )
        disp.plot(
            cmap="Blues",
            colorbar=False,
            xticks_rotation=45 if lev in ["acquisition", "task"] else None,
        )
        plt.savefig(source.parent / f"confusion_matrix_{lev}.png", bbox_inches="tight", dpi=300)
        plt.savefig(source.parent / f"confusion_matrix_{lev}.pdf", bbox_inches="tight")
        plt.close()

    log.info(f"{metrics_df}")

    out_filename = source.with_name(source.stem + "_metrics.csv")
    metrics_df.to_csv(out_filename, sep=",")
    log.info(f"Saved results to: {out_filename}")


def add_correct_frames(df, metrics_df, gt_corridor, level, show_correct):
    for lev in level:
        correct = dict()
        acc = dict()

        for t in gt_corridor:
            task_df = df[df["task"] == t]
            correct_tmp = len(task_df[task_df[lev] == task_df[f"pred_{lev}"]])
            correct.update({t: correct_tmp})
            acc.update({t: correct_tmp / len(task_df)})

        if show_correct:
            metrics_df[f"{lev}_correct"] = metrics_df["gt_corridor"].map(correct)

        metrics_df[f"{lev}_acc"] = metrics_df["gt_corridor"].map(acc)


if __name__ == "__main__":
    main()
