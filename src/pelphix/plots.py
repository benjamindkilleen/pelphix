import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import click
from pathlib import Path
import logging

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
    plot_sequence_predictions(source)


def plot_sequence_predictions(source: Path):
    source = Path(source)
    df = pd.read_csv(source)
    # collapsing s1 predictions
    df.loc[df["pred_task"] == "s1_left", "pred_task"] = "s1"
    df.loc[df["pred_task"] == "s1_right", "pred_task"] = "s1"

    plt.rcParams.update({"font.size": 16})

    # names in csv
    level = ["task", "activity", "acquisition", "frame"]

    # names in paper
    names = ["Corridor", "Activity", "View", "Frame"]

    unique = get_unique_vals(df, level)
    # unique = dict(
    #     task=np.array(["s1", "s2", "teardrop_left", "teardrop_right", "ramus_left", "ramus_right"]),
    #     activity=np.array(["position_wire", "insert_wire", "insert_screw"]),
    #     acquisition=np.array(
    #         [
    #             "ap",
    #             "lateral",
    #             "inlet",
    #             "outlet",
    #             "oblique_right",
    #             "oblique_left",
    #             "teardrop_left",
    #             "teardrop_right",
    #         ]
    #     ),
    #     frame=np.array(["fluoro-hunting", "assessment"]),
    # )
    # log.debug(f"Unique values: {unique}")
    # exit()

    palette = "Spectral"
    colors = set_colors(unique, palette)
    nan_color = "w"
    add_colors(df, unique, level, colors, nan_color)

    out_filename = source.with_suffix(".png")
    size = (12, 6)
    bar_height = 0.2
    bar_space = 0.3
    build_figure(df, level, names, size, bar_height, bar_space, out_filename)
    log.info(f"Saved results to: {out_filename}")


def get_unique_vals(df, levels):
    unique = dict()
    for lev in levels:
        labels = np.unique(df[[lev, f"pred_{lev}"]].astype(str).values)
        unique.update({lev: labels})

    return unique


def set_colors(unique, palette, spread=4, offset=4):
    n_classes = len(np.concatenate([v for v in unique.values()]))
    n_colors = spread * n_classes  # spread color range

    base_colors = sns.color_palette(palette, n_colors).as_hex()
    colors = dict()
    for i, k in enumerate(unique):
        start = i * offset
        step = len(base_colors) // len(unique[k])
        colors[k] = base_colors[start::step]

    # colors = dict(
    #     (k, sns.color_palette(palette, len(v)).as_hex()) for k, v in unique.items()
    # )

    return colors


def add_colors(df, unique, level, colors, nan_color):
    for lev in level:
        gt_colors = []
        pred_colors = []
        for i in range(len(df)):
            gt = df[lev][i]
            pred = df[f"pred_{lev}"][i]

            if pd.isnull(gt):
                gt_color = nan_color
            else:
                gt_color = colors[lev][list(unique[lev]).index(gt)]

            if pd.isnull(pred):
                pred_color = nan_color
            else:
                pred_color = colors[lev][list(unique[lev]).index(pred)]

            gt_colors.append(gt_color)
            pred_colors.append(pred_color)

        df[f"gt_{lev}_colors"] = gt_colors
        df[f"pred_{lev}_colors"] = pred_colors


def build_figure(df, level, names, size, bar_height, bar_space, output):
    fig, ax = plt.subplots(len(level), 1, figsize=size)
    fig.tight_layout(h_pad=0.2)

    for i in range(len(df)):
        for idx, lev in enumerate(level):
            if len(level) == 1:
                ax.barh(0, 1, bar_height, color=df[f"gt_{lev}_colors"][i], left=i, label=df[lev][i])
                ax.barh(
                    bar_space,
                    1,
                    bar_height,
                    color=df[f"pred_{lev}_colors"][i],
                    left=i,
                    label=df[f"pred_{lev}"][i],
                )
            else:
                ax[idx].barh(
                    0, 1, bar_height, color=df[f"gt_{lev}_colors"][i], left=i, label=df[lev][i]
                )
                ax[idx].barh(
                    bar_space,
                    1,
                    bar_height,
                    color=df[f"pred_{lev}_colors"][i],
                    left=i,
                    label=df[f"pred_{lev}"][i],
                )

    for idx, ax in enumerate(plt.gcf().get_axes()):
        ax.set_title(names[idx].capitalize())

        handles, labels = ax.get_legend_handles_labels()
        new_labels = []
        for i, l in enumerate(labels):
            if l == "nan":
                handles.pop(i)
            else:
                new_labels.append(l.replace("_", " ").title())

        by_label = dict(zip(new_labels, handles))
        if len(by_label.keys()) > 3:
            ax.legend(
                by_label.values(),
                by_label.keys(),
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                ncol=2,
            )
        else:
            ax.legend(
                by_label.values(), by_label.keys(), loc="center left", bbox_to_anchor=(1, 0.5)
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_xlim([0, len(df)])
        ax.set_yticks([0, bar_space], ["Ground Truth", "Predicted"])

        if idx != (len(level) - 1):
            ax.spines["bottom"].set_visible(False)
            ax.set_xticks([])
        else:
            ax.set_xlabel("Frame Number")

    fig.savefig(output, bbox_inches="tight", dpi=300)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
