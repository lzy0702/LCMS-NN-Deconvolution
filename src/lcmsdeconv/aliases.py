"""Name aliases: text lcmsdeconv shows in place of other text in everything it writes or prints.

A pair OLD=NEW (``123456=保密项目1``) makes the names, paths and texts lcmsdeconv takes from a run,
a method or the command line show NEW wherever they hold OLD: the run's name and source path, its
traces' names, units and metadata, the method's name and description and its other texts that
reach an output, the paths printed, and the warnings and errors printed. The pairs come from the
user at run time (``--alias OLD=NEW``, ``--alias-file FILE``, the file the environment variable
``LCMSDECONV_ALIAS_FILE`` names, or this module's API); none is built in, and no output records
an OLD or a pair's text.

The aliases are applied at the source: to the :class:`~lcmsdeconv.core.model.Run` and
:class:`~lcmsdeconv.core.method.Method` objects as they are read (:meth:`Aliases.alias_run`,
:meth:`Aliases.alias_method`), never to formatted output, so that numbers formatted later (an area
of 1234560, a mass of 123456.7) stay as they are. Text that exists only as text, a Python warning,
a log record or an error quoting a file's path, is aliased as text (:meth:`Aliases.console`).

Matching (:meth:`Aliases.apply`) is case-sensitive, with the pairs and the names in Unicode's
composed form (NFC). At each position the longest OLD wins, in a single pass: a NEW is never
matched again. An OLD that begins with a digit does not match where a digit comes right before
it, and one that ends with a digit does not match where a digit comes right after it, so 123456
matches in "LC-123456-S", "LC123456", "123456_2" and "123456.2" but not in "1234567" or
"0123456". An OLD or NEW may not hold an invisible or control character (a zero-width space, a
byte order mark, NUL, ...): such an OLD would match nothing, and the pair is refused.
"""

from __future__ import annotations

import codecs
import contextlib
import copy
import dataclasses
import html
import json
import logging
import os
import re
import sys
import threading
import traceback
import unicodedata
import warnings
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

#: the environment variable naming an alias file; read by the command line, the desktop
#: application and :meth:`Aliases.from_env`
ENV_VAR = "LCMSDECONV_ALIAS_FILE"

#: what :meth:`Aliases.leaks` shows in a file's name in place of each OLD found there
MASK = "***"

#: ``quant.purity_signal``'s keywords; any other value names a trace or a peak table
_SIGNAL_KEYWORDS = frozenset({"uv", "tic", "deic"})

#: the keywords of the method settings aliased only where they hold free text (a value that is,
#: or would become, a keyword is left as it is, so that the method works as before)
_INSTRUMENT_KINDS = frozenset({"tof", "orbitrap", "fticr"})
_WEIGHTINGS = frozenset({"none", "1/x", "1/x2", "1/x^2"})
_SWITCH_WORDS = frozenset({"on", "off", "true", "false", "yes", "no", "1", "0"})

#: the Unicode categories an OLD or NEW may not hold: control (NUL, TAB, ...), format (zero-width
#: space, byte order mark, direction marks), surrogate, line and paragraph separators
_INVISIBLE = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})

#: an alias file's line ends (LF, CRLF and a lone CR; nothing else ends a line)
_LINE_END = re.compile(r"\r\n|\r|\n")
_LINE_END_BYTES = re.compile(rb"\r\n|\r|\n")

#: a data URI (the report's embedded images): no text to check
_DATA_URI = re.compile(r"data:[\w.+/-]+;base64,[A-Za-z0-9+/=\s]*")


class AliasError(ValueError):
    """Alias pairs that cannot be used. The message names the file and the line (or the
    ``--alias``) but never quotes the line, which holds the text being hidden."""


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _invisible(text: str) -> str | None:
    """The first invisible or control character of ``text`` as "U+XXXX"; None where it holds
    none."""
    for ch in text:
        if unicodedata.category(ch) in _INVISIBLE:
            return f"U+{ord(ch):04X}"
    return None


def _pair_problem(old: str, new: str, empty: tuple[str, str] = ("nothing before '='",
                                                                 "nothing after '='")
                  ) -> str | None:
    """What is wrong with a stripped pair (``empty``: what to say of an empty OLD and NEW),
    never quoting it; None where it can be used."""
    if not old:
        return empty[0]
    if not new:
        return empty[1]
    for side, text in (("OLD", old), ("NEW", new)):
        code = _invisible(text)
        if code is not None:
            return f"{side} holds {code}, an invisible or control character"
    return None


def _pattern(pairs: Mapping[str, str]) -> re.Pattern:
    """One pattern for every OLD, the longest first (Python tries alternatives in order, so at
    each position the longest OLD that matches there wins), with the digit rules."""
    alternatives = []
    for old in sorted(pairs, key=lambda s: (-len(s), s)):
        alt = re.escape(old)
        if old[0].isdecimal():
            alt = r"(?<!\d)" + alt
        if old[-1].isdecimal():
            alt += r"(?!\d)"
        alternatives.append(alt)
    return re.compile("|".join(alternatives))


def _number_part(text: str, i: int, j: int) -> bool:
    """Whether text[i:j] is part of a number: a digit right before or after it, or a '.' or ','
    with a digit beyond it on either side ("0.123456", "123456.7", "123456,5")."""
    before = text[i - 1] if i > 0 else ""
    after = text[j] if j < len(text) else ""
    if before.isdecimal() or after.isdecimal():
        return True
    if after in (".", ",") and j + 1 < len(text) and text[j + 1].isdecimal():
        return True
    return before in (".", ",") and i >= 2 and text[i - 2].isdecimal()


def _number_or_switch(text: str) -> bool:
    """Whether a timed event's text value works as a number or an on/off switch."""
    if text.strip().lower() in _SWITCH_WORDS:
        return True
    try:
        float(text)
    except ValueError:
        return False
    return True


def _event_key(text: str) -> str:
    """A timed event's name as lcmsdeconv.chrom.events.build_schedule compares it."""
    return text.strip().lower().replace(" ", "_")


def unique_names(names: list[str]) -> list[str]:
    """``names`` with each repeat made unique: the first keeps its name, the later ones get
    " (2)", " (3)", ... (a suffix no name of the list holds already), so that names aliasing
    made equal stay apart and no data is lost."""
    taken = set(names)
    used: set[str] = set()
    out = []
    for name in names:
        if name in used:
            k = 2
            while f"{name} ({k})" in taken:
                k += 1
            name = f"{name} ({k})"
            taken.add(name)
        used.add(name)
        out.append(name)
    return out


class Aliases:
    """OLD=NEW pairs, applied to names, paths and texts (see the module's docstring).

    Without pairs it changes nothing: every method returns what it was given. ``len()`` is the
    number of pairs; its repr shows that number, never a pair. Build one with :meth:`parse`
    ("OLD=NEW" texts), :meth:`from_file`, :meth:`from_env` or :meth:`collect` (the command line's
    sources and precedence), and combine them with :meth:`merged` or ``|`` (a later pair for one
    OLD replaces an earlier one).
    """

    __slots__ = ("_pairs", "_pattern")

    def __init__(self, pairs: Mapping[str, str] | Iterable[tuple[str, str]] = ()):
        items = pairs.items() if isinstance(pairs, Mapping) else pairs
        merged: dict[str, str] = {}
        for i, (old, new) in enumerate(items, 1):
            old, new = _nfc(str(old).strip()), _nfc(str(new).strip())
            problem = _pair_problem(old, new, ("OLD is empty", "NEW is empty"))
            if problem is not None:
                raise AliasError(f"alias pair {i}: {problem}")
            merged.pop(old, None)  # a later pair for one OLD replaces an earlier one
            merged[old] = new
        self._pairs = merged
        self._pattern = _pattern(merged) if merged else None

    # ------------------------------------------------------------------ sources
    @classmethod
    def parse(cls, pairs: str | Iterable[str | tuple[str, str]],
              source: str = "--alias") -> Aliases:
        """Pairs given as "OLD=NEW" texts (split at the first '=', whitespace around OLD and NEW
        stripped) or (OLD, NEW) tuples. AliasError names ``source`` and the pair's number where a
        text has no '=', an empty OLD or NEW, or an invisible or control character in either."""
        if isinstance(pairs, str):
            pairs = [pairs]
        out = []
        for i, item in enumerate(pairs, 1):
            if isinstance(item, str):
                old, sep, new = item.partition("=")
                if not sep:
                    raise AliasError(f"{source} #{i}: no '=' (expected OLD=NEW)")
            else:
                old, new = item
            old, new = str(old).strip(), str(new).strip()
            problem = _pair_problem(old, new)
            if problem is not None:
                raise AliasError(f"{source} #{i}: {problem}")
            out.append((old, new))
        return cls(out)

    @classmethod
    def from_file(cls, path: str | os.PathLike, known: Aliases | None = None) -> Aliases:
        """The pairs of an alias file: UTF-8 (a byte order mark is accepted, at the start of any
        line too, as where two files were joined), LF, CRLF or CR line ends, one OLD=NEW per line
        split at the first '=', whitespace around OLD and NEW stripped; blank lines and lines
        starting with '#' are skipped.

        AliasError for a file that is missing or cannot be read, is not UTF-8 (UTF-16 included),
        or has a line without '=', with an empty OLD or NEW, or with an invisible or control
        character in either: the message names the file (shown with the pairs ``known`` and
        those read before the line applied) and the line's number, never the line itself.
        """
        path = Path(path)
        pairs: list[tuple[str, str]] = []

        def error(reason: str, line: int | None = None) -> AliasError:
            shown = (known or cls()).merged(cls(pairs)).apply(str(path))
            where = f"alias file {shown}" + ("" if line is None else f", line {line}")
            return AliasError(f"{where}: {reason}")

        def line_of(data: bytes, offset: int) -> int:
            return len(_LINE_END_BYTES.split(data[:offset]))

        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise error("not found") from None
        except IsADirectoryError:
            raise error("is a directory") from None
        except OSError as exc:  # its text would quote the path
            raise error(f"cannot be read ({exc.strerror or type(exc).__name__})") from None
        body = data[len(codecs.BOM_UTF8):] if data.startswith(codecs.BOM_UTF8) else data
        if b"\x00" in body:  # valid UTF-8, but text in UTF-16 holds them
            raise error("not UTF-8 (it holds NUL bytes, as UTF-16 text does)",
                        line_of(body, body.index(b"\x00"))) from None
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise error("not UTF-8", line_of(body, exc.start)) from None
        for n, line in enumerate(_LINE_END.split(text), 1):
            line = line.strip().removeprefix("\ufeff").strip()
            if not line or line.startswith("#"):
                continue
            old, sep, new = line.partition("=")
            if not sep:
                raise error("no '=' (expected OLD=NEW)", n)
            old, new = old.strip(), new.strip()
            problem = _pair_problem(old, new)
            if problem is not None:
                raise error(problem, n)
            pairs.append((old, new))
        return cls(pairs)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None,
                 known: Aliases | None = None) -> Aliases:
        """The pairs of the file LCMSDECONV_ALIAS_FILE names (in ``environ``, default
        ``os.environ``); none where it is unset or empty. AliasError as for :meth:`from_file`
        (``known``: pairs to show the file's path with there)."""
        value = (os.environ if environ is None else environ).get(ENV_VAR, "")
        if not value.strip():
            return cls()
        try:
            return cls.from_file(value, known=known)
        except AliasError as exc:
            raise AliasError(f"{ENV_VAR}: {exc}") from None

    @classmethod
    def collect(cls, files: Iterable[str | os.PathLike] = (),
                pairs: Iterable[str] = (), environ: Mapping[str, str] | None = None) -> Aliases:
        """The command line's aliases: the pairs of LCMSDECONV_ALIAS_FILE first, then those of
        each of ``files`` (``--alias-file``) in turn, then ``pairs`` ("OLD=NEW", ``--alias``); a
        later pair for one OLD replaces an earlier one. ``pairs`` are read first, so that a
        failing file's path is shown with them too."""
        given = cls.parse(pairs)
        out = cls.from_env(environ, known=given)
        for f in files:
            out = out.merged(cls.from_file(f, known=out.merged(given)))
        return out.merged(given)

    def merged(self, *others: Aliases) -> Aliases:
        """These pairs, then each of ``others``': a later pair for one OLD replaces an earlier
        one."""
        pairs = dict(self._pairs)
        for other in others:
            for old, new in other._pairs.items():
                pairs.pop(old, None)
                pairs[old] = new
        return Aliases(pairs)

    def __or__(self, other: Aliases) -> Aliases:
        if not isinstance(other, Aliases):
            return NotImplemented
        return self.merged(other)

    @property
    def pairs(self) -> dict[str, str]:
        """The pairs, OLD: NEW (a copy), in NFC."""
        return dict(self._pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def __bool__(self) -> bool:
        return bool(self._pairs)

    def __repr__(self) -> str:
        n = len(self._pairs)
        return f"Aliases({n} {'pair' if n == 1 else 'pairs'})"

    # ------------------------------------------------------------------ text
    def apply(self, text: Any) -> Any:
        """``text`` with every OLD replaced by its NEW (the matching rules of the module's
        docstring). A text holding no OLD comes back as it is (the very object, whether or not
        it is in NFC); anything but a str comes back unchanged."""
        if self._pattern is None or not isinstance(text, str) or not text:
            return text
        nfc = _nfc(text)
        out = self._pattern.sub(self._new, nfc)
        return text if out == nfc else out

    def _new(self, match: re.Match) -> str:
        return self._pairs[match.group(0)]

    def _hits(self, text: str) -> int:
        """How many times an OLD occurs in ``text`` (NFC) where it is not part of a number
        (:func:`_number_part`), counted for each OLD."""
        n = 0
        for old in self._pairs:
            i = text.find(old)
            while i >= 0:
                j = i + len(old)
                if not _number_part(text, i, j):
                    n += 1
                i = text.find(old, j)
        return n

    def count(self, text: str) -> int:
        """How many times an OLD occurs in ``text``, not counting those that are part of a
        number: with a digit, or a '.' or ',' and a digit, right before or after them."""
        if not self._pairs or not isinstance(text, str) or not text:
            return 0
        return self._hits(_nfc(text))

    def mask(self, text: str) -> str:
        """``text`` with every occurrence of every OLD (part of a number or not) shown as
        MASK."""
        if not self._pairs or not text:
            return text
        olds = sorted(self._pairs, key=lambda s: (-len(s), s))
        return re.sub("|".join(re.escape(o) for o in olds), MASK, _nfc(text))

    def leaks(self, paths: Iterable[str | os.PathLike], root: str | os.PathLike,
              text: bool = True) -> list[tuple[str, int]]:
        """The files of ``paths`` that still hold an OLD outside a number (:meth:`count`), in
        their name relative to ``root`` or (``text``) in their text: (that name with each OLD
        masked, the number of occurrences) for each. The text checked is a JSON file's decoded
        strings, an HTML file's text with its entities decoded and its data URIs (embedded
        images) left out, any other file's text. A file that cannot be read counts by its name
        only, as does a directory."""
        root = Path(root)
        out = []
        for p in paths:
            p = Path(p)
            try:
                name = p.relative_to(root).as_posix()
            except ValueError:
                name = p.name
            n = self.count(name) + (self.count(_file_text(p)) if text else 0)
            if n:
                out.append((self.mask(name), n))
        return out

    # ------------------------------------------------------------------ objects
    def alias_value(self, value: Any) -> Any:
        """``value`` with its strings aliased, within dicts (values, not keys), lists and tuples
        too; anything else (numbers, arrays) as it is. The very object where nothing changes,
        else a copy: ``value`` itself is never changed."""
        return self._value(value)[0]

    def _value(self, v: Any) -> tuple[Any, bool]:
        if isinstance(v, str):
            new = self.apply(v)
            return new, new is not v
        if isinstance(v, dict):
            done = {k: self._value(x) for k, x in v.items()}
            if any(changed for _x, changed in done.values()):
                return {k: x for k, (x, _c) in done.items()}, True
            return v, False
        if isinstance(v, (list, tuple)) and not hasattr(v, "_fields"):
            items = [self._value(x) for x in v]
            if any(changed for _x, changed in items):
                kind = list if isinstance(v, list) else tuple
                return kind(x for x, _c in items), True
            return v, False
        return v, False

    def alias_chromatogram(self, chrom, name: str | None = None):
        """A copy of the trace ``chrom`` with its name (``name`` where given), unit and metadata
        aliased; its arrays are shared. Without pairs, ``chrom`` itself."""
        if not self:
            return chrom
        return dataclasses.replace(chrom, name=self.apply(chrom.name) if name is None else name,
                                   unit=self.apply(chrom.unit),
                                   meta=dict(self.alias_value(chrom.meta)))

    def _trace_names(self, chroms: Mapping[str, Any]) -> tuple[list[str], list[str]]:
        """The keys of ``chroms`` aliased and made unique (:func:`unique_names`), and the names
        its traces get (a trace named as its key gets that key), in order."""
        keys = unique_names([self.apply(k) for k in chroms])
        names = [key if c.name == k else self.apply(c.name)
                 for (k, c), key in zip(chroms.items(), keys)]
        return keys, names

    def alias_chromatograms(self, chroms: Mapping[str, Any]) -> dict[str, Any]:
        """Traces by name (``Run.chromatograms``), their names (the keys and the objects')
        aliased: where aliasing makes two names equal, the later ones get " (2)", " (3)", ...
        (:func:`unique_names`), so no trace is lost. Without pairs, ``chroms`` itself."""
        if not self:
            return chroms
        keys, names = self._trace_names(chroms)
        return {key: self.alias_chromatogram(c, name)
                for c, key, name in zip(chroms.values(), keys, names)}

    def _spectrum(self, s):
        scan_id = self.apply(s.scan_id)
        meta, changed = self._value(s.meta)
        if scan_id is s.scan_id and not changed:
            return s
        return dataclasses.replace(s, scan_id=scan_id, meta=meta)

    def alias_run(self, run):
        """A copy of ``run`` with its name, source path, traces (names, units, metadata: see
        :meth:`alias_chromatograms`), metadata strings and the scans' IDs and metadata strings
        aliased. Its arrays, and the values within its metadata that hold no OLD, are shared,
        and a scan with nothing to alias is the run's own Spectrum object; ``run`` itself is
        left as it was. Without pairs, ``run`` itself."""
        if not self:
            return run
        return dataclasses.replace(run, spectra=[self._spectrum(s) for s in run.spectra],
                                   chromatograms=self.alias_chromatograms(run.chromatograms),
                                   name=self.apply(run.name), source=self.apply(run.source),
                                   meta=dict(self.alias_value(run.meta)))

    def _setting(self, value: Any, keywords: frozenset, key=None) -> Any:
        """A keyword setting's ``value`` aliased where it is free text: where neither it nor its
        aliased form is one of ``keywords`` (compared as ``key`` makes them), so that the setting
        works as before."""
        if not isinstance(value, str):
            return value
        new = self.apply(value)
        if new is value:
            return value
        k = key or (lambda s: s)
        return value if k(value) in keywords or k(new) in keywords else new

    def _event_value(self, value: Any) -> Any:
        """A timed event's value aliased where it is free text: neither it nor its aliased form a
        number or an on/off switch."""
        if not isinstance(value, str):
            return value
        new = self.apply(value)
        if new is value or _number_or_switch(value) or _number_or_switch(new):
            return value
        return new

    def _trace_references(self, method, run) -> tuple[Any, Any]:
        """``method``'s ``acquisition.uv_channel`` and ``quant.purity_signal`` as they must read
        for ``run`` aliased (:meth:`alias_method`): naming the same trace, or peak table, of the
        aliased run as of ``run``, or none where they named none."""
        channel, signal = method.acquisition.uv_channel, method.quant.purity_signal
        traces = list(run.chromatograms.values())
        _keys, names = self._trace_names(run.chromatograms)
        uvs = [i for i, c in enumerate(traces) if c.kind == "uv"]
        if not uvs:  # the channel names nothing; the signal names a TIC or a deconvolved trace's
            return None, signal  # table, which are named after aliasing, as before
        # lcmsdeconv.process: the first UV trace so named, else the first UV trace
        picked = next((i for i in uvs if traces[i].name == channel), None)
        channel = None if picked is None else names[picked]  # None: the first, as before
        picked = uvs[0] if picked is None else picked
        if signal == "uv":
            return channel, signal
        if signal == traces[picked].name:  # the UV trace's peak table
            return channel, names[picked]
        if signal == names[picked]:  # another table, or none: aliased, it would name the UV's
            return channel, None
        return channel, signal

    def alias_method(self, method, run=None):
        """A copy of ``method`` with its texts that reach an output aliased: its name and
        description, ``report.title``, ``instrument.kind``, ``quant.calibration.amount_unit`` and
        ``weighting``, each integration setting's ``area_unit`` and its timed events' names and
        text values. A setting whose value, or aliased value, is one of its keywords (``tof``,
        ``1/x``, ``baseline_now``, ``on``, ...) or a number is left as it is, so that the method
        works as before; so are its other settings (adduct names, numbers, the model's path).

        The method's references to the run's traces, ``acquisition.uv_channel`` and a
        ``quant.purity_signal`` other than uv, tic or deic, must name the same traces in the
        aliased run as in the run read. Pass that run (``run``, before :meth:`alias_run`): each
        is then set to the aliased name of the trace it names there, and one naming no trace
        still names none, even where aliasing made two trace names equal. Without ``run`` they
        are aliased as text, which names another trace where it did. ``method`` itself is not
        changed; without pairs, it is returned.
        """
        if not self or method is None:
            return method
        from .chrom.events import ALL_EVENTS

        m = copy.deepcopy(method)
        m.name = self.apply(m.name)
        m.description = self.apply(m.description)
        m.report.title = self.apply(m.report.title)
        if m.instrument is not None:
            m.instrument.kind = self._setting(m.instrument.kind, _INSTRUMENT_KINDS)
        calibration = m.quant.calibration
        calibration.amount_unit = self.apply(calibration.amount_unit)
        calibration.weighting = self._setting(calibration.weighting, _WEIGHTINGS)
        events = frozenset(ALL_EVENTS)
        for settings in m.integration.values():
            settings.area_unit = self.apply(settings.area_unit)
            for ev in settings.timed_events:
                ev.event = self._setting(ev.event, events, _event_key)
                ev.value = self._event_value(ev.value)
        if run is not None:
            m.acquisition.uv_channel, m.quant.purity_signal = self._trace_references(method, run)
        else:
            m.acquisition.uv_channel = self.apply(m.acquisition.uv_channel)
            if m.quant.purity_signal not in _SIGNAL_KEYWORDS:
                m.quant.purity_signal = self.apply(m.quant.purity_signal)
        return m

    # ------------------------------------------------------------------ console
    def console(self) -> contextlib.AbstractContextManager:
        """A context within which Python warnings, log records, the traceback of an exception
        no one catches (in a thread too) and that of one leaving the context are printed with
        the aliases applied to their text (as text: they quote file paths and names the readers
        saw before any object was aliased). No change without pairs, or within another such
        context."""
        return _console(self)


# ---------------------------------------------------------------------- files checked
def _file_text(path: Path) -> str:
    """The text :meth:`Aliases.leaks` checks in a file; "" where it cannot be read."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:  # its non-ASCII text is written \\uXXXX
            return json.dumps(json.loads(text), ensure_ascii=False)
        except ValueError:
            return text
    if suffix in (".html", ".htm"):
        return html.unescape(_DATA_URI.sub("", text))
    return text


# ---------------------------------------------------------------------- console
_console_depth = 0

#: the attribute that marks an exception which left a console() context with its aliases
_ESCAPED = "_lcmsdeconv_aliases"


def _alias_record(aliases: Aliases, record: logging.LogRecord) -> None:
    """Alias a log record's message, exception and stack text in place (once per record)."""
    if getattr(record, "_lcmsdeconv_aliased", False):
        return
    record._lcmsdeconv_aliased = True
    try:
        message = record.getMessage()
    except Exception:  # the handler reports a record it cannot format
        return
    record.msg, record.args = aliases.apply(message), None
    if record.exc_info and not record.exc_text:
        record.exc_text = logging.Formatter().formatException(record.exc_info)
    if record.exc_text:
        record.exc_text = aliases.apply(record.exc_text)
    if record.stack_info:
        record.stack_info = aliases.apply(record.stack_info)


def _chain(exc: BaseException | None) -> Iterator[BaseException]:
    """``exc`` and the exceptions it was raised from or while handling."""
    seen: set[int] = set()
    todo = [exc]
    while todo:
        e = todo.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        yield e
        todo += [e.__cause__, e.__context__]


def _format(aliases: Aliases, kind, value, tb) -> str:
    return aliases.apply("".join(traceback.format_exception(kind, value, tb)))


def _write(text: str) -> None:
    stream = sys.stderr
    if stream is not None:
        with contextlib.suppress(Exception):
            stream.write(text)
            stream.flush()


def _escaped_hook(previous):
    """A sys.excepthook printing an exception that left a console() context (marked with its
    aliases), or was raised from or while handling one, with those aliases applied; any other
    with ``previous``."""
    def hook(kind, value, tb):
        aliases = next((getattr(e, _ESCAPED) for e in _chain(value) if hasattr(e, _ESCAPED)),
                       None)
        if aliases is None:
            return previous(kind, value, tb)
        _write(_format(aliases, kind, value, tb))

    hook._lcmsdeconv_escaped = True
    return hook


@contextlib.contextmanager
def _console(aliases: Aliases) -> Iterator[None]:
    global _console_depth
    if not aliases or _console_depth:
        yield
        return

    def show_warning(message, category, filename, lineno, file=None, line=None):
        text = aliases.apply(warnings.formatwarning(message, category, filename, lineno, line))
        stream = sys.stderr if file is None else file
        if stream is not None:
            with contextlib.suppress(OSError):
                stream.write(text)

    def excepthook(kind, value, tb):
        _write(_format(aliases, kind, value, tb))

    def thread_hook(args):
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread is not None else threading.get_ident()
        _write(aliases.apply(f"Exception in thread {name}:\n")
               + _format(aliases, args.exc_type, args.exc_value, args.exc_traceback))

    factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = factory(*args, **kwargs)
        _alias_record(aliases, record)
        return record

    saved = (warnings.showwarning, sys.excepthook, threading.excepthook)
    warnings.showwarning, sys.excepthook, threading.excepthook = (show_warning, excepthook,
                                                                  thread_hook)
    logging.setLogRecordFactory(record_factory)
    _console_depth += 1
    escaped: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        escaped = exc
        raise
    finally:
        _console_depth -= 1
        if logging.getLogRecordFactory() is record_factory:
            logging.setLogRecordFactory(factory)
        if warnings.showwarning is show_warning:
            warnings.showwarning = saved[0]
        if threading.excepthook is thread_hook:
            threading.excepthook = saved[2]
        if sys.excepthook is excepthook:
            sys.excepthook = saved[1]
            # an exception leaving the context is printed after it, where no one catches it:
            # marked, it is printed aliased then (and any other as before)
            if escaped is not None and not isinstance(escaped, (SystemExit, GeneratorExit)):
                with contextlib.suppress(Exception):
                    setattr(escaped, _ESCAPED, aliases)
                if not getattr(sys.excepthook, "_lcmsdeconv_escaped", False):
                    sys.excepthook = _escaped_hook(sys.excepthook)
