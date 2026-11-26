"""Command-line interface for lcmsdeconv."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import click

from . import __version__

# ------------------------------------------------------------------ name aliases
#: where a command keeps its aliases (lcmsdeconv.aliases.Aliases) in the context's shared meta,
#: the alias options given, and the notes about them (AliasWarning texts)
_ALIASES = "lcmsdeconv.aliases"
_ALIAS_ARGS = "lcmsdeconv.alias_args"
_ALIAS_NOTES = "lcmsdeconv.alias_notes"


def _alias_callback(ctx, param, value):
    """Both alias options are eager, so they are read before any other parameter (whose errors
    can quote a path): once both are in, the command's aliases are put together, the pairs of
    the file LCMSDECONV_ALIAS_FILE names first, then --alias-file's, then --alias's. A note about
    them (an alias file holding no pair) is printed on stderr."""
    if ctx.resilient_parsing:
        return value
    given = ctx.meta.setdefault(_ALIAS_ARGS, {})
    given[param.name] = value
    if {"alias_file", "alias"} <= given.keys():
        import warnings

        from .aliases import AliasError, Aliases, AliasWarning

        caught: list = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", AliasWarning)
                ctx.meta[_ALIASES] = Aliases.collect(files=given["alias_file"],
                                                     pairs=given["alias"])
        except AliasError as exc:  # its message shows the paths aliased already
            error = click.ClickException(str(exc))
            error.lcmsdeconv_aliased = True
            raise error from None
        finally:
            notes = ctx.meta.setdefault(_ALIAS_NOTES, [])
            for w in caught:
                if issubclass(w.category, AliasWarning):
                    notes.append(str(w.message))
                    click.echo(f"WARNING: {w.message}", err=True)
                else:  # as it would have been shown
                    warnings.showwarning(w.message, w.category, w.filename, w.lineno, w.file,
                                         w.line)
    return value


def _alias_options(f):
    """--alias-file and --alias, for the commands that write or print names of runs, traces or
    methods (with :class:`_AliasedCommand`)."""
    f = click.option(
        "--alias", "alias", multiple=True, metavar="OLD=NEW", is_eager=True, expose_value=False,
        callback=_alias_callback,
        help="Show NEW in place of OLD in the names, paths and texts taken from the run, the "
             "method and the command line, in every output and message (repeatable; after "
             "--alias-file: a later pair for one OLD wins). Taken by process, gui, report, "
             "evaluate and synth.")(f)
    return click.option(
        "--alias-file", "alias_file", multiple=True, metavar="FILE", is_eager=True,
        expose_value=False, callback=_alias_callback,
        help="Read OLD=NEW pairs from FILE: UTF-8, one pair per line, '#' starts a comment line "
             "(repeatable; after the file the environment variable LCMSDECONV_ALIAS_FILE names, "
             "before --alias).")(f)


def _aliases():
    """The current command's aliases (none where it takes no alias options)."""
    from .aliases import Aliases

    ctx = click.get_current_context(silent=True)
    found = ctx.meta.get(_ALIASES) if ctx is not None else None
    return found if found is not None else Aliases()


def _env_aliases():
    """The pairs of the file LCMSDECONV_ALIAS_FILE names, for an error raised before a command's
    alias options are read (none where they cannot be read: that error comes later)."""
    from .aliases import AliasError, Aliases

    try:
        return Aliases.from_env()
    except AliasError:
        return Aliases()


class _AliasedGroup(click.Group):
    """The command group: its own errors (an unknown command, which quotes what was given, a run
    file put before the command, say) show the pairs of LCMSDECONV_ALIAS_FILE."""

    def parse_args(self, ctx, args):
        try:
            return super().parse_args(ctx, args)
        except click.ClickException as exc:
            exc.message = _env_aliases().apply(exc.message)
            raise

    def resolve_command(self, ctx, args):
        try:
            return super().resolve_command(ctx, args)
        except click.ClickException as exc:
            exc.message = _env_aliases().apply(exc.message)
            raise


class _AliasedCommand(click.Command):
    """A command whose messages show its aliases (``--alias``, ``--alias-file``, the file
    LCMSDECONV_ALIAS_FILE names): click's errors about its parameters (a path that does not
    exist), its own errors, and, while it runs, Python warnings, log records and the traceback
    of an unexpected error, each aliased as text. Without aliases, it is click's Command."""

    def parse_args(self, ctx, args):
        try:
            return super().parse_args(ctx, args)
        except click.ClickException as exc:
            if getattr(exc, "lcmsdeconv_aliased", False):
                raise
            aliases = ctx.meta.get(_ALIASES)
            if aliases is None:  # an error before the alias options were read
                aliases = _env_aliases()
            exc.message = aliases.apply(exc.message)
            raise

    def invoke(self, ctx):
        aliases = ctx.meta.get(_ALIASES)
        if not aliases:
            return super().invoke(ctx)
        # click's own exceptions end the command: raised after the console, not through it
        # (an exception leaving it would keep its hook in sys.excepthook)
        stop = None
        with aliases.console():
            try:
                return super().invoke(ctx)
            except click.ClickException as exc:
                exc.message = aliases.apply(exc.message)
                stop = exc
            except (click.exceptions.Exit, click.exceptions.Abort) as exc:
                stop = exc
            except Exception:
                click.echo(aliases.apply(traceback.format_exc()), err=True, nl=False)
        if stop is not None:
            raise stop
        ctx.exit(1)


class _GuiCommand(_AliasedCommand):
    """`gui`: an alias file or pair that cannot be used, and a note about them, are also shown
    in a dialog, as the application may be started without a console (a shortcut to pythonw),
    where nothing printed is seen."""

    def parse_args(self, ctx, args):
        try:
            return super().parse_args(ctx, args)
        except click.ClickException as exc:
            if getattr(exc, "lcmsdeconv_aliased", False):  # from the alias options
                _alias_dialogs(ctx.meta.get(_ALIAS_NOTES, []), exc.message)
            raise


def _alias_dialogs(notes, error=None) -> None:
    """Show notes about the aliases, and an error that keeps the application from starting, in
    dialogs (where the desktop application can be loaded)."""
    if not notes and error is None:
        return
    try:
        from .gui.app import alias_dialog
    except ImportError:  # no PySide6: printed on stderr only
        return
    for note in notes:
        alias_dialog(note)
    if error is not None:
        alias_dialog(error, fatal=True)


def _check_outputs(aliases, written, outdir: Path, others: bool = True) -> None:
    """Warn, on stderr, of each file written that still holds an alias's OLD outside a number
    (in its name relative to ``outdir`` or its text), naming the file (each OLD masked) and the
    count, never the text found. With ``others``, likewise of what else ``outdir`` holds, at any
    depth (an earlier run's outputs, say: :meth:`lcmsdeconv.aliases.Aliases.leaks_in`). Never
    fails."""
    found: list[tuple[str, int]] = []
    left: list[tuple[str, int]] = []
    try:  # the check never fails the command
        found = aliases.leaks(written, outdir)
    except Exception:
        pass
    try:
        if others:
            left = aliases.leaks_in(outdir, exclude=written)
    except Exception:
        pass

    def times(n):
        return f"{n} {'occurrence' if n == 1 else 'occurrences'}"

    for name, n in found:
        click.echo(f"WARNING: {name} still holds {times(n)} of an alias's OLD text", err=True)
    for name, n in left:
        click.echo(f"WARNING: {name}, which this run did not write, holds {times(n)} of an "
                   "alias's OLD text", err=True)


@click.group(cls=_AliasedGroup, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__)
def main():
    """Neural-network deconvolution and quantitation for LC-ESI-MS of macromolecules."""


@main.command(cls=_AliasedCommand)
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--method", "-m", default=None, help="Method YAML file or bundled method name.")
@click.option("--out", "-o", default="results", type=click.Path(file_okay=False),
              help="Output directory.")
@click.option("--model", default=None, type=click.Path(dir_okay=False),
              help="ONNX charge model (defaults to the bundled model).")
@click.option("--uv-csv", default=None, type=click.Path(exists=True, dir_okay=False),
              help="UV trace as a two-column CSV/TXT when it is not inside the mzML/mzMLb.")
@click.option("--waters-function", default=None, type=click.IntRange(min=1),
              help="Waters MS function to read, the N of 'function=N' in the scan IDs "
                   "(default: the function with the most MS1 scans, which leaves out the "
                   "lockspray function).")
@click.option("--report/--no-report", default=True, help="Write an HTML report.")
@click.option("--quiet", is_flag=True)
@_alias_options
def process(input_file, method, out, model, uv_csv, waters_function, report, quiet):
    """Process an mzML, mzMLb (or Thermo .raw) run: deconvolve, link, integrate and quantify.

    With alias pairs (--alias, --alias-file, the file the environment variable
    LCMSDECONV_ALIAS_FILE names), every output and message shows NEW in place of OLD in the
    run's, its traces' and the method's names and in the paths printed, and the number of pairs
    in effect is printed first; the files written are then checked for any OLD left, and so is
    what else --out holds, at any depth (an earlier run's outputs: give a fresh --out). See
    docs/usage.md, "Name aliases".
    """
    from .core.method import Method, rt_window_text
    from .io.mzml import read_mzml, read_uv_csv
    from .io.results import write_outputs
    from .process import process_run

    aliases = _aliases()
    shown = aliases.apply  # a path or name as printed
    # the method as loaded and the run as read are aliased together once the run is read (the
    # method's trace references follow the traces' aliased names): everything after shows the
    # aliased names only
    meth = Method.load(method)
    if aliases and not quiet:  # so the pairs can be seen to be in effect
        click.echo(f"name aliases: {len(aliases)} {'pair' if len(aliases) == 1 else 'pairs'}")
    if not quiet:
        click.echo(f"reading {shown(str(input_file))}")
    if str(input_file).lower().endswith(".raw"):
        if waters_function is not None:
            raise click.UsageError("--waters-function is for Waters runs in mzML or mzMLb")
        from .io.thermo import read_thermo_raw

        run = read_thermo_raw(input_file)
    else:
        from .io.vocabularies import VocabularyError

        try:
            run = read_mzml(input_file, function=waters_function)
        # no scans of that function, or an mzMLb file refused; or a saved vocabulary damaged
        except (ValueError, VocabularyError) as exc:
            raise click.ClickException(str(exc)) from exc
    if uv_csv:
        ch = read_uv_csv(uv_csv)
        run.chromatograms[ch.name] = ch
    run, meth = aliases.alias_run(run), aliases.alias_method(meth, run)  # both with the run read
    if not quiet:
        click.echo(f"{len(run.spectra)} MS frames, polarities {run.polarities}")
        if "waters_functions" in run.meta:
            click.echo(f"Waters function {run.meta['function']} read "
                       f"(MS1 scans by function: {run.meta['waters_functions']})")
        if run.meta.get("lockspray"):
            click.echo(f"{len(run.meta['lockspray'])} lockspray scans kept"
                       + ("" if meth.acquisition.lock_mass_mz is not None
                          else " (not used: the method sets no acquisition.lock_mass_mz)"))

    def progress(msg):
        if not quiet:
            click.echo(f"  {msg}")

    # the lock-mass correction and the analysis window are applied here, once
    result = process_run(run, meth, model_path=model, progress=progress)
    result.alias_pairs = len(aliases)
    if not quiet and result.lock_mass:
        from .io.lockmass import lock_mass_text

        click.echo(f"lock mass {result.lock_mass['reference_mz']:.4f}: "
                   f"{lock_mass_text(result.lock_mass)}")
    if not quiet and result.rt_window:
        click.echo(f"analysis window: {rt_window_text(result.rt_window)}")
    outdir = Path(out)
    written = write_outputs(result, outdir, report=report, plots=meth.report.plots)
    if aliases:
        _check_outputs(aliases, written, outdir)
    if report and not quiet:
        click.echo(f"report: {shown(str(outdir / 'report.html'))}")
    if not quiet:
        click.echo(f"{len(result.species)} species, {len(result.impurities)} rows in the impurity table")
        for w in result.warnings:
            click.echo(f"WARNING: {w}")
    click.echo(shown(str(outdir / "results.json")))


def _write_csvs(result, outdir: Path) -> list[Path]:
    """species.csv and the peak tables' CSVs (:func:`lcmsdeconv.io.results.write_csvs`)."""
    from .io.results import write_csvs

    return write_csvs(result, outdir)


def _csv_names(names: list[str]) -> dict[str, str]:
    """The peak tables' CSV name parts (:func:`lcmsdeconv.io.results.csv_names`)."""
    from .io.results import csv_names

    return csv_names(names)


@main.command(cls=_AliasedCommand)
@click.option("--preset", default="protein",
              type=click.Choice(["protein", "oligo", "polymer", "small_molecule", "mixture"]))
@click.option("--out", "-o", default="runs/demo", type=click.Path(file_okay=False))
@click.option("--peaks", default=3, help="Number of chromatographic peaks.")
@click.option("--minutes", default=6.0, help="Run length in minutes.")
@click.option("--rate", default=2.0, help="MS scan rate in Hz.")
@click.option("--saturate", is_flag=True, help="Include an ESI-saturated peak.")
@click.option("--switch-polarity", is_flag=True, help="Alternate positive and negative frames.")
@click.option("--seed", default=0)
@_alias_options
def synth(preset, out, peaks, minutes, rate, saturate, switch_polarity, seed):
    """Generate a synthetic LC-MS run (mzML plus ground truth).

    Its names are lcmsdeconv's own; the aliases apply to the path it prints.
    """
    import numpy as np

    from .io.mzml_writer import write_mzml
    from .io.results import save_json
    from .io.vocabularies import VocabularyError
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
    try:
        path = write_mzml(run, outdir / "run.mzML")
    except VocabularyError as exc:
        raise click.ClickException(str(exc)) from exc
    save_json(truth.to_dict(), outdir / "truth.json")
    click.echo(f"wrote {_aliases().apply(str(path))} ({len(run.spectra)} frames) and truth.json")


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
@click.option("--arch", default="unet", type=click.Choice(["unet", "ladder"]))
@click.option("--singly-charged", default="keep", type=click.Choice(["keep", "drop", "ignore"]))
@click.option("--resume", is_flag=True, help="Continue a paused or interrupted run.")
@click.option("--restart", is_flag=True, help="Discard a paused run's state and start again.")
@click.option("--out", default="models/charge_unet.pt", type=click.Path(dir_okay=False))
def train(config, model_size, epochs, train_len, val_len, batch_size, workers, max_seconds, arch,
          singly_charged, resume, restart, out):
    """Train the charge network on synthetic data (scripts/train_charge_net.py has more)."""
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
                      batch_size=batch_size, num_workers=workers, max_seconds=max_seconds, out=out,
                      arch=arch, singly_charged=singly_charged, resume=resume, restart=restart)
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
@click.option("--strict", is_flag=True,
              help="exit with code 2 if the ONNX model does not match the checkpoint")
def export_onnx_cmd(checkpoint, out, length, verify, strict):
    """Export a trained checkpoint to ONNX for deployment."""
    from .nn.export import export_onnx, verify_parity

    out = out or str(Path(checkpoint).with_suffix(".onnx"))
    p = export_onnx(checkpoint, out, length=length)
    click.echo(f"wrote {p}")
    if verify:
        res = verify_parity(checkpoint, p)
        click.echo(json.dumps(res, indent=2))
        if strict and not res["ok"]:
            click.echo("the ONNX model does not match the checkpoint", err=True)
            raise SystemExit(2)


@main.command(cls=_AliasedCommand)
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--truth", default=None, type=click.Path(exists=True, dir_okay=False),
              help="truth.json from `lcmsdeconv synth` for scoring.")
@click.option("--method", "-m", default=None)
@click.option("--model", default=None, type=click.Path(dir_okay=False))
@_alias_options
def evaluate(run_dir, truth, method, model):
    """Score processing output against synthetic ground truth.

    Its scores hold no names; the aliases apply to the warnings and errors it prints.
    """
    from .evaluate import evaluate_directory

    res = evaluate_directory(Path(run_dir), Path(truth) if truth else None, method, model)
    click.echo(json.dumps(res, indent=2))


@main.command(cls=_AliasedCommand)
@click.argument("results_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--open", "open_it", is_flag=True, help="Open the report in a browser.")
@_alias_options
def report(results_dir, open_it):
    """Show where the HTML report is (and optionally open it).

    With alias pairs, the path is shown aliased and the report is checked for any OLD left (a
    report written without them is not rewritten).
    """
    aliases = _aliases()
    p = Path(results_dir) / "report.html"
    if not p.exists():
        click.echo(f"no report at {aliases.apply(str(p))}", err=True)
        sys.exit(1)
    if aliases:
        _check_outputs(aliases, [p], p.parent, others=False)
    click.echo(aliases.apply(str(p)))
    if open_it:
        import webbrowser

        webbrowser.open(p.resolve().as_uri())


@main.command(cls=_GuiCommand)
@click.option("--run", "run_file", default=None, type=click.Path(exists=True, dir_okay=False),
              help="mzML or mzMLb run to open.")
@click.option("--method", "-m", default=None)
@click.option("--waters-function", default=None, type=click.IntRange(min=1),
              help="Waters MS function of the --run file to read (default: as for process).")
@_alias_options
def gui(run_file, method, waters_function):
    """Launch the desktop application.

    With alias pairs, the application shows NEW in place of OLD in every name, path, table,
    legend and dialog of its own, and in the reports it exports (the system's file dialogs show
    your folders as they are); its status bar shows the number of pairs in effect.
    """
    if waters_function is not None and not run_file:
        raise click.UsageError("--waters-function applies to the --run file")
    from .gui.app import launch

    _alias_dialogs(click.get_current_context().meta.get(_ALIAS_NOTES, []))
    launch(run_file, method, waters_function=waters_function, aliases=_aliases())


@main.command("methods")
def methods_cmd():
    """List the bundled method templates."""
    from .core.method import bundled_method_dir, bundled_methods

    for name in bundled_methods():
        click.echo(name)
    click.echo(f"\n({bundled_method_dir()})")


if __name__ == "__main__":
    main()
