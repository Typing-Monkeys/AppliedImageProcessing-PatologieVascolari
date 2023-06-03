import pandas as pd
import argparse
from os import walk, sep, makedirs

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.transform import dodge
from bokeh.io import save, export_png
from bokeh.palettes import Dark2_5 as palette
import itertools

SAVE_FN = save
SAVE_EXT = "html"


def _calculate_data_for_plots(dfs: dict[str, pd.DataFrame],
                              metric: str) -> dict[str, float]:
    tot_res = {}
    indici = list(dfs['P18'].describe().keys()[1:])
    for indice in indici:
        res = []
        for name, df in dfs.items():
            tmp = df.describe().transpose()[metric].transpose()
            res.append((name, tmp[indice]))
        res = [x[1] for x in sorted(res, key=lambda y: y[0])]
        tot_res[indice] = res

    return tot_res


def make_charts(data_path: str, dest_path: str):
    dfs = {}
    for (dirpath, _, filenames) in walk(data_path):
        for filename in filenames:
            if filename == "metrics.csv":
                name = dirpath.split(sep)[-1]
                dfs[name] = pd.read_csv(sep.join([dirpath, filename]))

    # PATIENTS CHARTS
    for name, df in dfs.items():
        t = df.iloc[:, 1:]
        colors = itertools.cycle(palette)
        p = figure(x_range=t['Filename'])
        p.title = name

        for feat, color in zip(t.iloc[:, 1:], colors):
            p.line(t['Filename'],
                   t[feat],
                   line_width=2,
                   color=color,
                   legend_label=f"{feat} {name}")
            p.legend.location = "top_left"
            p.legend.click_policy = "mute"
            p.xaxis.major_label_orientation = "vertical"

        SAVE_FN(p, filename=sep.join([dest_path, f"{name}.{SAVE_EXT}"]))

    # METRICS CHARTS
    statistiche = list(dfs['P18'].describe().transpose().keys())
    for statistica in statistiche:
        names_ordered = sorted(dfs.keys())
        act_values = _calculate_data_for_plots(dfs, statistica)
        data = {
            'x': names_ordered,
            'DCI': act_values['DCI'],
            'TI': act_values['TI'],
            'Em': act_values['Em'],
            'Bpn': act_values['Bpn'],
        }

        source = ColumnDataSource(data=data)
        p = figure(x_range=names_ordered, title=statistica.upper(), height=350)

        p.vbar(x=dodge('x', -0.37, range=p.x_range),
               top='DCI',
               source=source,
               width=0.2,
               color="#c9d9d3",
               legend_label="DCI")

        p.vbar(x=dodge('x', -0.12, range=p.x_range),
               top='TI',
               source=source,
               width=0.2,
               color="#718dbf",
               legend_label="TI")

        p.vbar(x=dodge('x', 0.12, range=p.x_range),
               top='Em',
               source=source,
               width=0.2,
               color="#e84d60",
               legend_label="Em")

        p.vbar(x=dodge('x', 0.37, range=p.x_range),
               top='Bpn',
               source=source,
               width=0.2,
               color="#6f7c00",
               legend_label="Bpn")

        SAVE_FN(p, filename=sep.join([dest_path, f"{statistica}.{SAVE_EXT}"]))


def main():
    parser = argparse.ArgumentParser(
        description='Auto generate charts from results data')
    parser.add_argument('source_dir',
                        type=str,
                        help='Folder where data are stored')

    parser.add_argument('dest_dir',
                        type=str,
                        help='Folder where to save charts')

    parser.add_argument('--png',
                        action="store_true",
                        default=False,
                        help='Export chart to PNG')

    args = parser.parse_args()

    global SAVE_FN
    global SAVE_EXT

    if args.png:
        SAVE_FN = export_png
        SAVE_EXT = "png"

    # take the first element from the generator
    # because I only need folders of depth 1 (source_dir subdirs)
    dirpath, dirnames, _ = next(walk(args.source_dir))
    for dirname in dirnames:
        dest_dir = sep.join([args.dest_dir, dirname])
        makedirs(dest_dir, exist_ok=True)

        make_charts(sep.join([dirpath, dirname]), dest_dir)


if __name__ == "__main__":
    main()
