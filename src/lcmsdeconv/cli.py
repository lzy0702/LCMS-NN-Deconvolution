"""Command-line interface for lcmsdeconv."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from . import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__)
def main():
    """Neural-network deconvolution and quantitation for LC-ESI-MS of macromolecules."""


@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--method", "-m", default=None, help="Method YAML file or bundled method name.")
@click.option("--out", "-o", default="results", type=click.Path(file_okay=False),
              help="Output directory.")
@click.option("--model", default=None, type=click.Path(dir_okay=False),
              help="ONNX charge model (defaults to the bundled model).")
@click.option("--uv-csv", default=None, type=click.Path(exists=True, dir_okay=False),
              help="UV trace as a two-column CSV/TXT when it is not inside the mzML.")
@click.option("--report/--no-report", default=True, help="Write an HTML report.")
@click.option("--quiet", is_flag=True)
def process(input_file, method, out, model, uv_csv, report, quiet):
    """Process an mzML run: deconvolve, link, integrate and quantify."""
    from .core.method import Method
    from .io.mzml import read_mzml, read_uv_csv
    from .io.results import save_json
    from .process import process_run

    meth = Method.load(method)
    if not quiet:
        click.echo(f"reading {input_file}")
    run = read_mzml(input_file)
    if uv_csv:
        ch = read_uv_csv(uv_csv)
        run.chromatograms[ch.name] = ch
    if not quiet:
        click.echo(f"{len(run.spectra)} MS frames, polarities {run.polarities}")

    def progress(msg):
        if not quiet:
            click.echo(f"  {msg}")

    result = process_run(run, meth, model_path=model, progress=progress)
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(result.summary(), outdir / "results.json")
    _write_csvs(result, outdir)
    if report:
        from .report.html import build_report

        p = build_report(result, outdir / "report.html", plots=meth.report.plots)
        if not quiet:
            click.echo(f"report: {p}")
    if not quiet:
        click.echo(f"{len(result.species)} species, {len(result.impurities)} rows in the impurity table")
        for w in result.warnings:
            click.echo(f"WARNING: {w}")
    click.echo(str(outdir / "results.json"))


def _write_csvs(result, outdir: Path):
    import pandas as pd

    if result.impurities:
        rows = []
        for r in result.impurities:
            rows.append({k: v for k, v in r.items() if k not in ("adducts", "flags")}
                        | {"adducts": "; ".join(f"{k} {v*100:.2f}%" for k, v in r["adducts"].items()),
                           "flags": "; ".join(r["flags"])})
        pd.DataFrame(rows).to_csv(outdir / "species.csv", index=False)
    for name, table in result.peak_tables.items():
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        table.to_dataframe().to_csv(outdir / f"peaks_{safe}.csv", index=False)


@main.command()
@click.option("--preset", default="protein",
              type=click.Choice(["protein", "oligo", "polymer", "small_molecule", "mixture"]))
@click.option("--out", "-o", default="runs/demo", type=click.Path(file_okay=False))
@click.option("--peaks", default=3, help="Number of chromatographic peaks.")
@click.option("--minutes", default=6.0, help="Run length in minutes.")
@click.option("--rate", default=2.0, help="MS scan rate in Hz.")
@click.option("--saturate", is_flag=True, help="Include an ESI-saturated peak.")
@click.option("--switch-polarity", is_flag=True, help="Alternate positive and negative frames.")
@click.option("--seed", default=0)
def synth(preset, out, peaks, minutes, rate, saturate, switch_polarity, seed):
    """Generate a synthetic LC-MS run (mzML plus ground truth)."""
    import numpy as np

    from .io.mzml_writer import write_mzml
    from .io.results import save_json
    from .synth.chromatography import generate_run
    from .synth.compounds import DEFAULT_CLASSES, ClassConfig
    from .synth.config import SynthConfig

    presets = {
        "protein": ([ClassConfig("peptide", (8000.0, 150000.0))], "rplc", 1),
        "oligo": ([ClassConfig("ps_rna", (4000.0, 25000.0)), ClassConfig("rna", (4000.0, 25000.0))], "iprp", -1),
        "polymer": ([ClassConfig("peg", (1000.0, 20000.0))], "polymer", 1),
        "small_molecule": ([ClassConfig("small_molecule", (150.0, 900.0))], "rplc", 1),
        "mixture": (list(DEFAULT_CLASSES), "rplc", 1),
    }
    classes, mode, polarity = presets[preset]
    cfg = SynthConfig(classes=classes, mode=mode, polarity=polarity)
    rng = np.random.default_rng(seed)
    run, truth = generate_run(cfg, rng, n_peaks=peaks, rt_range=(0.5, 0.5 + minutes),
                              scan_rate_hz=rate, polarity_switching=switch_polarity,
                              esi_saturation=saturate)
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    path = write_mzml(run, outdir / "run.mzML")
    save_json(truth.to_dict(), outdir / "truth.json")
    click.echo(f"wrote {path} ({len(run.spectra)} frames) and truth.json")


@main.command()
@click.option("--config", default=None, type=click.Path(exists=True, dir_okay=False),
              help="Training config YAML.")
@click.option("--model-size", default="proof", type=click.Choice(["proof", "small", "full"]))
@click.option("--epochs", default=10)
@click.option("--train-len", default=2000)
@click.option("--val-len", default=200)
@click.option("--batch-size", default=4)
@click.option("--workers", default=2)
@click.option("--max-seconds", default=None, type=float)
@click.option("--out", default="models/charge_unet.pt", type=click.Path(dir_okay=False))
def train(config, model_size, epochs, train_len, val_len, batch_size, workers, max_seconds, out):
    """Train the charge network on synthetic data."""
    import yaml

    from .nn.train import TrainConfig
    from .nn.train import train as run_train
    from .synth.config import SynthConfig

    kw = {}
    synth_cfg = SynthConfig()
    if config:
        data = yaml.safe_load(Path(config).read_text()) or {}
        kw = data.get("train", {})
        for k, v in (data.get("synth") or {}).items():
            if hasattr(synth_cfg, k):
                setattr(synth_cfg, k, v)
    cfg = TrainConfig(model_size=model_size, epochs=epochs, train_len=train_len, val_len=val_len,
                      batch_size=batch_size, num_workers=workers, max_seconds=max_seconds, out=out)
    for k, v in kw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    _model, history = run_train(cfg, synth_cfg)
    click.echo(json.dumps(history[-1] if history else {}, indent=2))


@main.command("export-onnx")
@click.argument("checkpoint", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "-o", default=None, type=click.Path(dir_okay=False))
@click.option("--length", default=32768)
@click.option("--verify/--no-verify", default=True)
def export_onnx_cmd(checkpoint, out, length, verify):
    """Export a trained checkpoint to ONNX for deployment."""
    from .nn.export import export_onnx, verify_parity

    out = out or str(Path(checkpoint).with_suffix(".onnx"))
    p = export_onnx(checkpoint, out, length=length)
    click.echo(f"wrote {p}")
    if verify:
        click.echo(json.dumps(verify_parity(checkpoint, p), indent=2))


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--truth", default=None, type=click.Path(exists=True, dir_okay=False),
              help="truth.json from `lcmsdeconv synth` for scoring.")
@click.option("--method", "-m", default=None)
@click.option("--model", default=None, type=click.Path(dir_okay=False))
def evaluate(run_dir, truth, method, model):
    """Score processing output against synthetic ground truth."""
    from .evaluate import evaluate_directory

    res = evaluate_directory(Path(run_dir), Path(truth) if truth else None, method, model)
    click.echo(json.dumps(res, indent=2))


@main.command()
@click.argument("results_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--open", "open_it", is_flag=True, help="Open the report in a browser.")
def report(results_dir, open_it):
    """Show where the HTML report is (and optionally open it)."""
    p = Path(results_dir) / "report.html"
    if not p.exists():
        click.echo(f"no report at {p}", err=True)
        sys.exit(1)
    click.echo(str(p))
    if open_it:
        import webbrowser

        webbrowser.open(p.resolve().as_uri())


@main.command()
@click.option("--run", "run_file", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--method", "-m", default=None)
def gui(run_file, method):
    """Launch the desktop application."""
    from .gui.app import launch

    launch(run_file, method)


@main.command("methods")
def methods_cmd():
    """List the bundled method templates."""
    from .core.method import bundled_method_dir, bundled_methods

    for name in bundled_methods():
        click.echo(name)
    click.echo(f"\n({bundled_method_dir()})")


if __name__ == "__main__":
    main()
