"""The PSI-MS and Unit Ontology vocabularies, read from the copies saved with the package.

pyteomics reads the cvParams of an mzML or mzMLb file with psims' PSI-MS vocabulary, and psims
writes mzML with PSI-MS and UO. Left to themselves, they fetch these from purl.obolibrary.org
(plain HTTP, no timeout): pyteomics 5 for every reader it makes (two per ``read_mzml``), psims
for the first writer of a process. lcmsdeconv hands them the copies saved in
``lcmsdeconv/data/cv`` instead (versions, sources and licences in its README.md;
``scripts/refresh_vocabularies.py`` replaces them), so reading and writing mzML make no network
request: each reader gets the vocabulary as its ``cv`` argument (:func:`reader_vocabulary`) and
each writer entries of its own for both, loaded from the saved copies
(:func:`make_mzml_writer`). No setting of psims or pyteomics is changed: psims' module-level
``obo_cache`` and its default ``<cvList>`` entries stay as they are for other code in the
process.
"""

from __future__ import annotations

import gzip
import threading
import warnings
import zlib
from pathlib import Path
from types import SimpleNamespace

PSI_MS = "http://purl.obolibrary.org/obo/ms/psi-ms.obo"
UO = "http://purl.obolibrary.org/obo/uo.obo"
#: the saved copy of each vocabulary, by the URI mzML files and psims name it by
SAVED = {PSI_MS: "psi-ms.obo.gz", UO: "uo.obo.gz"}
CV_DIR = Path(__file__).resolve().parent.parent / "data" / "cv"

_parsed: dict[str, object] = {}
_lock = threading.RLock()
#: terms met in files read that the saved vocabularies lack, each warned of once a process
_unknown: dict[str, object] = {}


class VocabularyError(RuntimeError):
    """A vocabulary saved with lcmsdeconv is missing or cannot be read: the installation is
    damaged. Not a ValueError, which psims takes as "not available" and lcmsdeconv's command line
    as a fault of the run file."""


def _path(uri: str) -> Path:
    if uri not in SAVED:
        raise ValueError(f"{uri}: no copy of this vocabulary is saved with lcmsdeconv, and "
                         "none is fetched")
    return CV_DIR / SAVED[uri]


def _parse(uri: str):
    from psims.controlled_vocabulary.controlled_vocabulary import ControlledVocabulary

    path = _path(uri)
    try:
        with gzip.open(path, "rb") as fh:
            # a vocabulary that imports another finds it here too, or not at all
            return ControlledVocabulary.from_obo(fh, import_resolver=vocabulary)
    # a file missing (OSError), cut short (EOFError), not gzip (OSError, zlib.error), or not an
    # OBO vocabulary with terms (ValueError)
    except (OSError, EOFError, zlib.error, ValueError) as exc:
        raise VocabularyError(
            f"{path}: this vocabulary, saved with lcmsdeconv, cannot be read "
            f"({type(exc).__name__}: {exc}); the installation is damaged: reinstall lcmsdeconv"
        ) from exc


def vocabulary(uri: str = PSI_MS):
    """The saved vocabulary ``uri`` names (PSI-MS by default) as a psims ControlledVocabulary,
    parsed on first use and kept for the process; threads asking at once get the same one.

    ValueError for any other vocabulary: psims and pyteomics take that as "not available", and
    nothing is fetched. :class:`VocabularyError` when the saved copy is missing or damaged.
    """
    cv = _parsed.get(uri)
    if cv is None:
        _path(uri)
        with _lock:
            cv = _parsed.get(uri)
            if cv is None:
                cv = _parsed[uri] = _parse(uri)
    return cv


class ReaderVocabulary:
    """The saved PSI-MS vocabulary as pyteomics' readers are given it (their ``cv`` argument).

    pyteomics looks up in it the type of each PSI-MS cvParam that carries a value, and the name
    of each unit given only by its accession, and fails the whole file on a term it does not
    hold: a term added to PSI-MS after the saved copy, or a wrong accession. A unit is looked up
    in the saved UO as well. Any other term is read as pyteomics reads a term it has no type for
    (a number where the value is one, else text; a unit by its accession), and named in a
    warning, once a process.
    """

    def __init__(self, cv):
        self.cv = cv

    @property
    def version(self) -> str:
        return self.cv.version

    def __getitem__(self, accession: str):
        try:
            return self.cv[accession]
        except KeyError:
            pass
        if str(accession).startswith("UO:"):
            try:
                return vocabulary(UO)[accession]
            except (KeyError, VocabularyError):  # a damaged UO copy shows when writing
                pass
        term = _unknown.get(accession)
        if term is None:
            new = SimpleNamespace(id=accession, name=accession, relationship=[])
            term = _unknown.setdefault(accession, new)
            if term is new:
                warnings.warn(
                    f"{accession} is not in the vocabularies saved with lcmsdeconv (PSI-MS "
                    f"{self.cv.version}), so it is read without them: a value as a number, or "
                    "as text where it is not one, a unit by its accession. "
                    "scripts/refresh_vocabularies.py saves newer vocabularies", stacklevel=2)
        return term


def reader_vocabulary() -> ReaderVocabulary:
    """The saved PSI-MS vocabulary for a pyteomics reader (:class:`ReaderVocabulary`)."""
    return ReaderVocabulary(vocabulary())


class SavedVocabularyResolver:
    """psims' vocabulary resolver interface (``load``, ``resolve``, ``fallback``), for
    ``MzMLWriter(vocabulary_resolver=...)``, over the saved vocabularies only: never remote.

    psims' own resolver, the module-level ``obo_cache``, downloads a vocabulary before falling
    back to the older copy psims bundles; this one serves the saved copies and refuses any other
    vocabulary (ValueError, which psims takes as "not available").
    """

    use_remote = False  # what psims reports as the writer's use_remote_controlled_vocabularies

    def load(self, uri: str):
        return vocabulary(uri)

    def resolve(self, uri: str):
        """A binary file over the saved copy of ``uri``."""
        return gzip.open(_path(uri), "rb")

    def fallback(self, uri: str):
        return None

    def __call__(self, uri: str):
        return self.resolve(uri)


def make_mzml_writer(fh, **kwargs):
    """psims' MzMLWriter over the file ``fh`` (``kwargs`` passed on) with the saved vocabularies.

    The writer is given entries of its own for PSI-MS and UO (with the ids, names and URIs of
    psims' defaults), loaded here from the saved copies, in place of psims' module-level default
    entries (``psims.mzml.components.default_cv_list``). Its ``<cvList>`` names the saved
    versions, and it never loads psims' entries, which the first writer of a process otherwise
    loads, or fetches, for every writer after it: they stay as they were, unloaded unless other
    code has loaded them (psims binds them to each writer it makes, this one too, which changes
    nothing a later writer sees). psims warns that the writer's own entries replace its
    defaults; that warning is not shown.
    """
    from psims.mzml import MzMLWriter
    from psims.mzml.components import default_cv_list
    from psims.xml import CV

    saved = SavedVocabularyResolver()
    own = []
    for default in default_cv_list:
        if default.uri in SAVED:
            cv = CV(full_name=default.full_name, id=default.id, uri=default.uri, resolver=saved)
            cv.vocabulary  # noqa: B018  (loaded now: the writer then never loads it itself)
            own.append(cv)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Replacing ", category=UserWarning)
        return MzMLWriter(fh, vocabularies=own, vocabulary_resolver=saved, **kwargs)
