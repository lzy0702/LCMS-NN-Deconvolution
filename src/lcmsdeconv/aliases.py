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
"0123456". An OLD may not hold an invisible, control or format character (a zero-width space, a
byte order mark, NUL, ...) nor a full-width or compatibility form of a character (the digits of
"１２３４５６", a no-break or ideographic space, a Roman numeral, a ligature): such an OLD would
not match the plain characters, and the pair is refused. A NEW may not hold a control character,
a line break or a byte order mark either, and neither may hold U+FFFD, which stands for a
character lost in a change of encoding.
"""

from __future__ import annotations

import builtins
import codecs
import contextlib
import copy
import dataclasses
import html
import json
import linecache
import logging
import os
import re
import stat
import sys
import threading
import traceback
import types
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

#: the Unicode categories an OLD may not hold: control (NUL, TAB, ...), format (zero-width space,
#: byte order mark, direction marks), surrogate, line and paragraph separators; a NEW may hold
#: format characters (Persian's zero-width non-joiner, an emoji's zero-width joiner, a soft
#: hyphen), which do not keep it from showing
_INVISIBLE = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})
_INVISIBLE_NEW = _INVISIBLE - {"Cf"}

#: an alias file's line ends (LF, CRLF and a lone CR; nothing else ends a line)
_LINE_END = re.compile(r"\r\n|\r|\n")
_LINE_END_BYTES = re.compile(rb"\r\n|\r|\n")

#: a character of three bytes in UTF-8 (as UTF-8 writes Chinese), which text in code page 936
#: seldom forms, though it often forms one of two bytes (路 is C2 B7 there)
_UTF8_THREE_BYTES = re.compile(rb"[\xe0-\xef][\x80-\xbf]{2}")

#: what an error about a file that is not UTF-8 tells the user to do
_SAVE_AS_UTF8 = "save it as UTF-8 (in Notepad: File > Save as, Encoding UTF-8)"

#: what an error about U+FFFD (the replacement character) says of it
_LOST = "which stands for a character lost in a change of encoding: type it again"

#: a data URI (the report's embedded images): no text to check
_DATA_URI = re.compile(r"data:[\w.+/-]+;base64,[A-Za-z0-9+/=\s]*")

#: the names `lcmsdeconv process` gives its outputs (what :meth:`Aliases.leaks_in` reads)
_OUTPUT_NAMES = ("results.json", "species.csv", "report.html")

#: the largest file whose text :meth:`Aliases.leaks_in` reads, in bytes
_TEXT_LIMIT = 50 * 2**20


class AliasError(ValueError):
    """Alias pairs that cannot be used. The message names the file and the line (or the
    ``--alias``) but never quotes the line, which holds the text being hidden."""


class AliasWarning(UserWarning):
    """Alias pairs that can be used but may not be what was meant (an alias file holding no
    pair). Like AliasError's, the message never quotes a line."""


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _invisible(text: str, categories: frozenset = _INVISIBLE) -> str | None:
    """The first character of ``text`` in one of ``categories`` (invisible or control
    characters) as "U+XXXX"; None where it holds none."""
    for ch in text:
        if unicodedata.category(ch) in categories:
            return f"U+{ord(ch):04X}"
    return None


def _compatibility(text: str) -> str | None:
    """The first character of ``text`` that Unicode's compatibility form (NFKC) writes otherwise
    than its composed form (NFC), a full-width digit, a no-break or ideographic space, a Roman
    numeral, ...; None where it holds none."""
    for ch in text:
        if unicodedata.normalize("NFKC", ch) != unicodedata.normalize("NFC", ch):
            return ch
    return None


def _typed_full_width(ch: str) -> bool:
    """Whether ``ch`` is a full-width form (as an input method types in full-width mode) or a
    space other than the plain one (a no-break or ideographic space)."""
    return "\uff01" <= ch <= "\uff60" or "\uffe0" <= ch <= "\uffe6" or (
        unicodedata.category(ch) == "Zs")


def _pair_problem(old: str, new: str, empty: tuple[str, str] = ("nothing before '='",
                                                                 "nothing after '='")
                  ) -> str | None:
    """What is wrong with a stripped pair (``empty``: what to say of an empty OLD and NEW),
    never quoting it; None where it can be used."""
    if not old:
        return empty[0]
    if not new:
        return empty[1]
    code = _invisible(old)
    if code is not None:
        return f"OLD holds {code}, an invisible or control character"
    ch = _compatibility(old)
    if ch is not None and _typed_full_width(ch):  # an input method in full-width mode, Word, ...
        return (f"OLD holds U+{ord(ch):04X}, a full-width or compatibility form of a character, "
                "which does not match the plain one: type it half-width (the input method in "
                "English or half-width mode)")
    if ch is not None:  # pasted from a PDF file, picked among an input method's candidates, ...
        return (f"OLD holds U+{ord(ch):04X}, a compatibility form of a character (a Roman "
                "numeral, a circled or superscript digit, a ligature, a radical pasted for a "
                "Chinese character, ...), which does not match the plain one: give an OLD without "
                "it, in plain characters")
    if "\ufffd" in old:
        return f"OLD holds U+FFFD, {_LOST}"
    code = _invisible(new, _INVISIBLE_NEW)
    if code is None and "\ufeff" in new:  # a byte order mark: where two files were joined
        code = "U+FEFF"
    if code is not None:
        return f"NEW holds {code}, an invisible or control character"
    if "\ufffd" in new:
        return f"NEW holds U+FFFD, {_LOST}"
    return None


def _file_parts(line: str) -> list[str]:
    """An alias file's line split where a file starting with a byte order mark was joined to it
    after a last line without a line end: at a byte order mark with a pair, a comment or nothing
    on each side. One elsewhere, a zero-width no-break space pasted within a pair or a comment,
    is left there: the pair is refused naming U+FEFF, and the comment stays one (a text holding
    '=' after it is read as a pair, never taken into the comment)."""
    pieces = line.split("\ufeff")
    parts = [pieces[0]]
    for piece in pieces[1:]:
        if all(not t or t.startswith("#") or "=" in t
               for t in (parts[-1].strip(), piece.strip())):
            parts.append(piece)
        else:
            parts[-1] += "\ufeff" + piece
    return parts


def _no_equals(line: str) -> str:
    """What to say of a pair without '=' (never quoting it)."""
    if "＝" in line:  # typed with an input method in full-width mode
        return ("no '=' (it holds U+FF1D, a full-width equals sign: type '=' half-width, the "
                "input method in English or half-width mode)")
    return "no '=' (expected OLD=NEW)"


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
        text has no '=', an empty OLD or NEW, an invisible or control character in either (a
        NEW may hold format characters but a byte order mark), U+FFFD in either, or a full-width
        or compatibility character in OLD."""
        if isinstance(pairs, str):
            pairs = [pairs]
        out = []
        for i, item in enumerate(pairs, 1):
            if isinstance(item, str):
                old, sep, new = item.partition("=")
                if not sep:
                    raise AliasError(f"{source} #{i}: {_no_equals(item)}")
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
        """The pairs of an alias file: UTF-8 (a byte order mark is accepted, and one between pairs
        or comments within a line starts a pair there, as where two files were joined, and so are
        end-of-file marks, Ctrl-Z, at its end, as cmd's ``copy a+b`` adds), LF, CRLF or CR line
        ends, one OLD=NEW per line split at the first '=', whitespace around OLD and NEW stripped;
        blank lines and lines starting with '#' are skipped. A file holding no pair gives none,
        with an AliasWarning, and so does one with a NEW holding '=' (two pairs on one line, where
        a file that did not end with a line end was joined to another or added to).

        AliasError for a file that is missing or cannot be read, is not UTF-8 (UTF-16 or an ANSI
        code page: the message says to save it as UTF-8; or lines added to it in another
        encoding), or has a line without '=', with an empty OLD or NEW, an invisible or control
        character in either (a NEW may hold format characters but a byte order mark), U+FFFD in
        either, or a full-width or compatibility character in OLD: the message names the file
        (shown with the pairs ``known`` and those read before the line applied) and the line's
        number, never the line itself.
        """
        path = Path(path)
        pairs: list[tuple[str, str]] = []

        def shown(text: str | None = None) -> str:
            return (known or cls()).merged(cls(pairs)).apply(str(path) if text is None else text)

        def error(reason: str, line: int | None = None) -> AliasError:
            where = f"alias file {shown()}" + ("" if line is None else f", line {line}")
            return AliasError(f"{where}: {reason}")

        def line_of(data: bytes, offset: int) -> int:
            return len(_LINE_END_BYTES.split(data[:offset]))

        def safely(test) -> bool:  # a test of a path that fails is a no
            try:
                return test()
            except (OSError, ValueError):
                return False

        try:
            data = path.read_bytes()
        except FileNotFoundError:
            twin = path.name + ".txt"  # Explorer hides the extension .txt
            if safely(lambda: path.with_name(twin).is_file()):
                raise error(f"not found, but {shown(twin)} is (Explorer hides the extension "
                            ".txt): give that name, or rename the file") from None
            raise error("not found") from None
        except OSError as exc:  # its text would quote the path
            if safely(path.is_dir):  # IsADirectoryError, PermissionError on Windows
                raise error("is a folder: give the alias file's own path (the folder's path "
                            "followed by the file's name)") from None
            raise error(f"cannot be read ({exc.strerror or type(exc).__name__})") from None
        if data[:2] in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE):
            raise error("UTF-16 (Notepad's 'UTF-16 LE' or 'UTF-16 BE', 'Unicode' in older "
                        "Notepad; the > and Out-File of Windows PowerShell 5.1), not UTF-8: "
                        f"{_SAVE_AS_UTF8}", 1)
        body = data[len(codecs.BOM_UTF8):] if data.startswith(codecs.BOM_UTF8) else data
        body = body.rstrip(b"\x1a")  # DOS end-of-file marks (cmd's copy a+b adds one)
        if b"\x00" in body:  # valid UTF-8, but text in UTF-16 holds them
            raise error(f"not UTF-8 (it holds NUL bytes, as UTF-16 text does): {_SAVE_AS_UTF8}",
                        line_of(body, body.index(b"\x00"))) from None
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            # UTF-8 before it (a byte order mark, or a character of three bytes, as UTF-8 writes
            # Chinese): lines added otherwise
            if data.startswith(codecs.BOM_UTF8) or _UTF8_THREE_BYTES.search(body, 0, exc.start):
                raise error("not UTF-8, though the text before it is (a line added in code page "
                            "936, with Windows PowerShell 5.1's Add-Content without -Encoding "
                            "UTF8 or cmd's echo >>?): type the pairs into a new file in Notepad "
                            "and save it as UTF-8", line_of(body, exc.start)) from None
            try:  # Chinese Windows' ANSI code page (Notepad's 'ANSI', PowerShell 5.1's Set-Content)
                body.decode("cp936")
                ansi = " (saved as ANSI, code page 936 or GBK?)"
            except UnicodeDecodeError:
                ansi = ""
            raise error(f"not UTF-8{ansi}: {_SAVE_AS_UTF8}", line_of(body, exc.start)) from None
        joined = []  # the lines with a NEW holding '='
        for n, line in enumerate(_LINE_END.split(text), 1):
            # a byte order mark starts a file: between pairs or comments within a line, a file
            # that did not end with a line end was joined to another there (cat a b, copy a+b),
            # and a pair starts there
            for part in _file_parts(line):
                part = part.strip()
                if not part or part.startswith("#"):
                    continue
                old, sep, new = part.partition("=")
                if not sep:
                    raise error(_no_equals(part), n)
                old, new = old.strip(), new.strip()
                problem = _pair_problem(old, new)
                if problem is not None:
                    raise error(problem, n)
                if "=" in new:
                    joined.append(n)
                pairs.append((old, new))
        if not pairs:
            warnings.warn(f"alias file {shown()} holds no pair (its lines are blank or comments): "
                          "no name is aliased", AliasWarning, stacklevel=2)
        if joined:
            lines = list(dict.fromkeys(joined))
            where = ", ".join(map(str, lines))
            warnings.warn(f"alias file {shown()}, line{'s' if len(lines) > 1 else ''} {where}: "
                          "NEW holds '=': where the line holds two pairs (a file without a line "
                          "end after its last pair, joined to another or added to), put a line "
                          "end between them", AliasWarning, stacklevel=2)
        return cls(pairs)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None,
                 known: Aliases | None = None) -> Aliases:
        """The pairs of the file LCMSDECONV_ALIAS_FILE names (in ``environ``, default
        ``os.environ``); none where it is unset or empty. The value may be in double quotes (as
        Windows' "Copy as path" gives a path) and start with ``~`` or hold environment variables
        (``%USERPROFILE%`` on Windows, ``$HOME`` elsewhere); a relative path is read from the
        current folder, and an error then shows the folder it was looked for in. AliasError as
        for :meth:`from_file` (``known``: pairs to show the file's path with there), and where
        LCMSDECONV_ALIAS_FILE is not set but a variable whose name is it with spaces around is (as
        cmd's ``set NAME = value`` makes)."""
        environ = os.environ if environ is None else environ
        value = environ.get(ENV_VAR)
        if value is None:
            spaced = next((k for k in environ if k != ENV_VAR and k.strip() == ENV_VAR), None)
            if spaced is not None:
                raise AliasError(
                    f"{ENV_VAR} is not set, but {spaced!r} is (its name with spaces): set it "
                    f"without spaces around '=' (in cmd: set \"{ENV_VAR}=C:\\path\\aliases.txt\")")
            return cls()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':  # no Windows path holds '"'
            value = value[1:-1].strip()
        if not value:
            return cls()
        path = Path(os.path.expandvars(os.path.expanduser(value)))
        try:
            return cls.from_file(path.absolute(), known=known)
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
        for old in self._pairs:  # the pattern matches only where an OLD is (the quicker test)
            if old in nfc:
                break
        else:
            return text
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

    def leaks_in(self, folder: str | os.PathLike, exclude: Iterable[str | os.PathLike] = (),
                 limit: int = 10_000) -> list[tuple[str, int]]:
        """What else ``folder`` holds, at any depth, that still holds an OLD outside a number
        (:meth:`count`): an earlier run's outputs, say, beside those of ``exclude`` (the files a
        run wrote there, for :meth:`leaks`), which are left out. Each file and folder is checked
        by its name, and each file named as ``lcmsdeconv process`` names its outputs
        (results.json, species.csv, report.html, peaks_*.csv) by its text too, read as
        :meth:`leaks` reads it: (its name relative to ``folder`` with each OLD masked, the number
        of occurrences in its own name and text) for each, sorted. At most ``limit`` entries are
        looked at, and no text above 50 MB is read."""
        if not self._pairs:
            return []
        def key(p) -> str:
            return os.path.normcase(os.path.abspath(p))

        root = Path(folder)
        skip = {key(p) for p in exclude}
        out = []
        seen = 0
        for top, dirs, files in os.walk(root):
            dirs.sort()
            for name, is_file in [(d, False) for d in dirs] + [(f, True) for f in sorted(files)]:
                seen += 1
                if seen > limit:
                    return sorted(out)
                p = Path(top) / name
                if key(p) in skip:
                    continue
                n = self.count(name)
                if is_file and (name in _OUTPUT_NAMES
                                or (name.startswith("peaks_") and name.endswith(".csv"))):
                    with contextlib.suppress(OSError):
                        st = p.stat()  # a regular file only: reading a FIFO or device blocks
                        if stat.S_ISREG(st.st_mode) and st.st_size <= _TEXT_LIMIT:
                            n += self.count(_file_text(p))
                if n:
                    out.append((self.mask(p.relative_to(root).as_posix()), n))
        return sorted(out)

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
        """A context within which what Python prints as text shows the aliases (as text: it
        quotes file paths and names the readers saw before any object was aliased). Warnings are
        passed on to the warnings hook in effect with their message, file name and source line
        aliased (so warnings.catch_warnings and logging.captureWarnings still get them); log
        records have their message, exception and stack text and other text fields (file,
        module, function and thread names, ``extra`` fields) aliased as they are made and as a
        handler takes them, a record holding no OLD being left as it is; the traceback of an
        exception no one catches (in a thread too), an exception Python ignores
        (sys.unraisablehook) and the message of a SystemExit leaving the context are aliased.

        An exception leaving the context is marked with the pairs, and printed aliased if no one
        catches it: sys.excepthook and threading.excepthook keep a hook that does so until it has
        printed every such exception (for good where one is caught). Contexts open at once,
        nested or in other threads, share the hooks, which apply the pairs of all of them (a
        later-opened one's pair for an OLD wins) and are restored when the last one closes. A
        thread started within must be joined within: one failing after it is printed as Python
        prints it. No change without pairs."""
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
#: guards the state below: console() contexts open and close in any thread
_lock = threading.RLock()


def _new_lock() -> None:
    """A new lock in a process forked while another thread held this one (it would never be
    released there)."""
    global _lock
    _lock = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_new_lock)

#: the console() contexts open now, in the order they were entered: (token, pairs)
_open: list[tuple[object, Aliases]] = []
#: the pairs the hooks apply, those of the open contexts merged; None while none is open
_in_effect: Aliases | None = None
#: the installation of the hooks in effect (a hook of another one passes everything on but the
#: exceptions marked as having left a context)
_installed: object | None = None
#: how many exceptions left a context and were not printed by these hooks yet (until then,
#: sys.excepthook and threading.excepthook keep a hook that prints them aliased)
_pending = 0

#: the attributes marking an exception which left a console() context (with its pairs), one
#: printed since, and a SystemExit whose message was aliased
_ESCAPED = "_lcmsdeconv_aliases"
_PRINTED = "_lcmsdeconv_printed"
_EXITED = "_lcmsdeconv_exit_aliased"
#: the attribute of a log record holding what console() set in it, by attribute
_RECORD = "_lcmsdeconv_aliased"
#: a log record's attributes _alias_record leaves to the rest of it or as they are: the logger's
#: name (which filters compare), the level, and what a formatter makes of the others
_RECORD_KEEP = frozenset({"name", "msg", "args", "levelname", "exc_text", "message", "asctime",
                          _RECORD})
#: exception groups (Python 3.11 on)
_GROUP = getattr(builtins, "BaseExceptionGroup", None)


def _shown_args(aliases: Aliases, args: Any) -> Any:
    """The ``args`` of a log record that cannot be formatted, aliased as logging's error report
    prints them (their reprs): a text aliased, anything else whose repr holds an OLD as that
    repr aliased; the very object where nothing changes."""
    def one(a):
        if isinstance(a, str):
            return aliases.apply(a)
        try:
            text = repr(a)
        except Exception:
            return a
        shown = aliases.apply(text)
        return a if shown is text else shown

    if isinstance(args, tuple):
        items = [one(a) for a in args]
        return args if all(x is a for x, a in zip(items, args)) else tuple(items)
    if isinstance(args, Mapping):
        values = {k: one(v) for k, v in args.items()}
        return args if all(values[k] is v for k, v in args.items()) else values
    return args if args is None else one(args)


def _alias_record(aliases: Aliases, record: logging.LogRecord) -> None:
    """Alias a log record's text in place where it holds an OLD: its message (``msg`` the
    aliased message and ``args`` None; for a record that cannot be formatted, ``msg`` and
    ``args``, which logging's error report prints), its exception text, and its other text
    attributes (stack, file, module, function and thread names, ``extra`` fields). A record
    holding no OLD is left as it is, and a text aliased once is not aliased again (a NEW may
    hold its OLD)."""
    state = vars(record)
    done = state.get(_RECORD) or {}

    def mine(attr: str) -> bool:
        return attr in done and state.get(attr) is done[attr]

    changes: dict[str, Any] = {}
    if not mine("msg"):
        try:
            message = record.getMessage()
        except Exception:  # logging reports it, printing msg and args
            msg, args = aliases.apply(record.msg), _shown_args(aliases, record.args)
            if msg is not record.msg or args is not record.args:
                changes.update(msg=msg, args=args)
        else:
            shown = aliases.apply(message)
            if shown is not message:
                changes.update(msg=shown, args=None)
    if not mine("exc_text"):
        text = record.exc_text
        if not text and record.exc_info and record.exc_info[0] is not None:
            # formatted here only to look for an OLD: the handler's formatter formats it
            text = logging.Formatter().formatException(record.exc_info)
        if text:
            shown = aliases.apply(text)
            if shown is not text:
                changes["exc_text"] = shown
    for attr, value in state.items():
        if isinstance(value, str) and attr not in _RECORD_KEEP and not mine(attr):
            shown = aliases.apply(value)
            if shown is not value:
                changes[attr] = shown
    if changes:
        state.update(changes)
        state[_RECORD] = {**done, **changes}


def _warning_shown(aliases: Aliases, message, filename, lineno, line):
    """A warning's message, file name and source line as console() passes them on: aliased
    where they hold an OLD (the message as a warning of its own class, where that takes one
    text), else as they were given."""
    text = str(message)
    shown = aliases.apply(text)
    if shown is not text:
        if isinstance(message, Warning):
            try:
                message = type(message)(shown)
            except Exception:
                message = shown
        else:
            message = shown
    name = aliases.apply(filename)
    source = line
    if source is None:  # as warnings.formatwarning reads it, under the file's real name
        try:
            source = linecache.getline(filename, lineno)
        except Exception:
            source = ""
    shown_source = aliases.apply(source)
    if name is not filename or shown_source is not source:
        line = shown_source
    return message, name, line


def _unraisable_text(args) -> str:
    """What Python prints for an exception it ignores (sys.unraisablehook's default)."""
    message = "Exception ignored in" if args.err_msg is None else args.err_msg
    if args.object is not None:
        try:
            shown = repr(args.object)
        except Exception:
            shown = "<object repr() failed>"
        head = f"{message}: {shown}\n"
    else:
        head = f"{message}\n"
    if args.exc_type is None:
        return head
    return head + "".join(traceback.format_exception(args.exc_type, args.exc_value,
                                                     args.exc_traceback))


def _alias_exit(exc: SystemExit, aliases: Aliases) -> None:
    """Alias the message a SystemExit prints (a text code: ``sys.exit("cannot read ...")``),
    once."""
    code = exc.code
    if code is None or isinstance(code, int) or getattr(exc, _EXITED, False):
        return
    text = str(code)
    shown = aliases.apply(text)
    with contextlib.suppress(Exception):
        if shown is not text:
            exc.code, exc.args = shown, (shown,)
        setattr(exc, _EXITED, True)


def _chain(exc: BaseException | None) -> Iterator[BaseException]:
    """``exc``, the exceptions it was raised from or while handling, and those of a group."""
    seen: set[int] = set()
    todo = [exc]
    while todo:
        e = todo.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        yield e
        todo += [e.__cause__, e.__context__]
        if _GROUP is not None and isinstance(e, _GROUP):
            todo += list(e.exceptions)


def _escape(exc: BaseException, aliases: Aliases) -> None:
    """Mark an exception leaving a console() context with its pairs, so that the exception hooks
    print it aliased where no one catches it, even after the context; alias the message of a
    SystemExit, which Python prints without them."""
    global _pending
    if isinstance(exc, SystemExit):
        _alias_exit(exc, _in_effect or aliases)
        return
    if isinstance(exc, GeneratorExit):
        return
    earlier = getattr(exc, _ESCAPED, None)
    fresh = not isinstance(earlier, Aliases) or getattr(exc, _PRINTED, False)
    try:
        setattr(exc, _ESCAPED, aliases.merged(earlier) if isinstance(earlier, Aliases)
                else aliases)
        setattr(exc, _PRINTED, False)
    except Exception:
        return
    if fresh:
        with _lock:
            _pending += 1


#: the code of console()'s hooks (that of _install's functions)
_HOOK_CODE: set[types.CodeType] = set()


def _mark(hook) -> tuple[object, Any] | None:
    """(installation, hook replaced) of a console() hook; None for any other hook. Told by its
    code, which functools.wraps does not copy onto a wrapper as it copies the attributes, and
    read from the function's own attributes, never with getattr: a Mock makes up any attribute
    asked of it."""
    if not isinstance(hook, types.FunctionType) or hook.__code__ not in _HOOK_CODE:
        return None
    own = hook.__dict__
    if "_lcmsdeconv_token" not in own:
        return None
    return own["_lcmsdeconv_token"], own["_lcmsdeconv_previous"]


def _restored(hook):
    """``hook`` without the console() hooks of closed contexts on top of it."""
    mark = _mark(hook)
    while mark is not None and mark[0] is not _installed:
        hook = mark[1]
        mark = _mark(hook)
    return hook


def _unhook_exceptions() -> None:
    sys.excepthook = _restored(sys.excepthook)
    threading.excepthook = _restored(threading.excepthook)


def _printed(marked: list[BaseException]) -> None:
    """Count exceptions that left a context as printed; once none is left, put back the
    exception hooks of contexts closed since."""
    global _pending
    with _lock:
        for e in marked:
            if not getattr(e, _PRINTED, True):
                with contextlib.suppress(Exception):
                    setattr(e, _PRINTED, True)
                    _pending -= 1
        if _pending <= 0 and _installed is None:
            _pending = 0
            _unhook_exceptions()


def _pairs_for(active: bool, value: BaseException | None) -> Aliases | None:
    """The pairs to print an exception with: those in effect (where ``active``), and those of
    the contexts it, or an exception of its chain, left (which are counted as printed)."""
    marked = [e for e in _chain(value) if isinstance(getattr(e, _ESCAPED, None), Aliases)]
    pairs = [getattr(e, _ESCAPED) for e in marked]
    in_effect = _in_effect
    if active and in_effect is not None:
        pairs.insert(0, in_effect)
    if marked:
        _printed(marked)
    if not pairs:
        return None
    return pairs[0] if len(pairs) == 1 else Aliases().merged(*pairs)


def _write(text: str) -> None:
    stream = sys.stderr
    if stream is not None:
        with contextlib.suppress(Exception):
            stream.write(text)
            stream.flush()


def _hook(function, token: object, previous):
    """``function``, a hook of the installation ``token`` put over ``previous``."""
    _HOOK_CODE.add(function.__code__)
    function._lcmsdeconv_token = token
    function._lcmsdeconv_previous = previous
    return function


def _install() -> None:
    """Put the hooks in (the first context opening; _lock held). Each passes on what holds no
    OLD to the hook it replaced, as it was given."""
    global _installed
    # the exception hooks kept for exceptions that left a closed context (_pending) are taken
    # off, not wrapped: these hooks print those exceptions too (no installation is in effect
    # here, so _restored takes off every one on top)
    previous_except = _restored(sys.excepthook)
    previous_thread = _restored(threading.excepthook)
    token = _installed = object()

    def pairs() -> Aliases | None:
        return _in_effect if _installed is token else None

    previous_warning = warnings.showwarning

    def show_warning(message, category, filename, lineno, file=None, line=None):
        aliases = pairs()
        if aliases is not None:
            message, filename, line = _warning_shown(aliases, message, filename, lineno, line)
        return previous_warning(message, category, filename, lineno, file, line)

    def except_hook(kind, value, tb):
        aliases = _pairs_for(_installed is token, value)
        if aliases is not None:
            text = "".join(traceback.format_exception(kind, value, tb))
            shown = aliases.apply(text)
            if shown is not text:
                _write(shown)
                return None
        return previous_except(kind, value, tb)

    def thread_hook(args):
        aliases = (None if args.exc_type is SystemExit
                   else _pairs_for(_installed is token, args.exc_value))
        if aliases is not None:
            name = args.thread.name if args.thread is not None else threading.get_ident()
            text = f"Exception in thread {name}:\n" + "".join(traceback.format_exception(
                args.exc_type, args.exc_value, args.exc_traceback))
            shown = aliases.apply(text)
            if shown is not text:
                _write(shown)
                return None
        return previous_thread(args)

    previous_unraisable = sys.unraisablehook

    def unraisable_hook(args):
        aliases = pairs()
        if aliases is not None:
            text = _unraisable_text(args)
            shown = aliases.apply(text)
            if shown is not text:
                _write(shown)
                return None
        return previous_unraisable(args)

    previous_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = previous_factory(*args, **kwargs)
        aliases = pairs()
        if aliases is not None:
            _alias_record(aliases, record)
        return record

    previous_handle = logging.Handler.handle

    def handle(self, record):  # a record made before the context, or changed since
        aliases = pairs()
        if aliases is not None:
            _alias_record(aliases, record)
        return previous_handle(self, record)

    warnings.showwarning = _hook(show_warning, token, previous_warning)
    sys.excepthook = _hook(except_hook, token, previous_except)
    threading.excepthook = _hook(thread_hook, token, previous_thread)
    sys.unraisablehook = _hook(unraisable_hook, token, previous_unraisable)
    logging.setLogRecordFactory(_hook(record_factory, token, previous_factory))
    logging.Handler.handle = _hook(handle, token, previous_handle)


def _uninstall() -> None:
    """Put the hooks back where they are still on top (the last context closing; _lock held),
    the exception hooks only once no exception that left a context waits to be printed."""
    global _installed
    _installed = None
    warnings.showwarning = _restored(warnings.showwarning)
    sys.unraisablehook = _restored(sys.unraisablehook)
    logging.setLogRecordFactory(_restored(logging.getLogRecordFactory()))
    logging.Handler.handle = _restored(logging.Handler.handle)
    if _pending <= 0:
        _unhook_exceptions()


def _merge(sets: list[Aliases]) -> Aliases:
    return sets[0] if all(s is sets[0] for s in sets) else Aliases().merged(*sets)


@contextlib.contextmanager
def _console(aliases: Aliases) -> Iterator[None]:
    global _in_effect
    if not aliases:
        yield
        return
    token = object()
    try:  # registered within, so that an interrupt (Ctrl-C) arriving meanwhile unregisters it
        with _lock:
            if not _open:
                _install()
            _open.append((token, aliases))
            _in_effect = _merge([a for _t, a in _open])
        yield
    except BaseException as exc:
        _escape(exc, aliases)
        raise
    finally:
        with _lock:
            _open[:] = [e for e in _open if e[0] is not token]
            _in_effect = _merge([a for _t, a in _open]) if _open else None
            if not _open:
                _uninstall()
