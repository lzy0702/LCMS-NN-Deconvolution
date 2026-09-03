"""Qt table models for peak, species and integration-event tables."""

from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt


class DictTableModel(QAbstractTableModel):
    """Read-only table over a list of dicts with a fixed column order."""

    def __init__(self, columns: list[tuple[str, str]], rows: list[dict] | None = None,
                 formats: dict[str, str] | None = None, parent=None):
        super().__init__(parent)
        self._columns = columns  # (key, header)
        self._rows = rows or []
        self._formats = formats or {}

    def set_rows(self, rows: list[dict]) -> None:
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def row_dict(self, row: int) -> dict:
        return self._rows[row] if 0 <= row < len(self._rows) else {}

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.ToolTipRole):
            return None
        key, _ = self._columns[index.column()]
        value = self._rows[index.row()].get(key)
        if value is None:
            return ""
        fmt = self._formats.get(key)
        if fmt and isinstance(value, (int, float)):
            try:
                return format(value, fmt)
            except (ValueError, TypeError):
                return str(value)
        if isinstance(value, dict):
            return ", ".join(f"{k} {v*100:.1f}%" for k, v in
                             sorted(value.items(), key=lambda x: -x[1])[:3])
        if isinstance(value, (list, tuple)):
            return "; ".join(str(v) for v in value)
        return str(value)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self._columns[section][1]
        return section + 1


class EventTableModel(QAbstractTableModel):
    """Editable table of timed integration events (time, event, value)."""

    HEADERS = ["Time (min)", "Event", "Value"]

    def __init__(self, events=None, parent=None):
        super().__init__(parent)
        self._events = list(events or [])

    def events(self):
        return list(self._events)

    def set_events(self, events) -> None:
        self.beginResetModel()
        self._events = list(events or [])
        self.endResetModel()

    def add_event(self, event) -> None:
        self.beginInsertRows(QModelIndex(), len(self._events), len(self._events))
        self._events.append(event)
        self.endInsertRows()

    def remove_row(self, row: int) -> None:
        if 0 <= row < len(self._events):
            self.beginRemoveRows(QModelIndex(), row, row)
            self._events.pop(row)
            self.endRemoveRows()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._events)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else 3

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.EditRole):
            return None
        ev = self._events[index.row()]
        return [f"{ev.time:.3f}", ev.event, "" if ev.value is None else str(ev.value)][index.column()]

    def setData(self, index, value, role=Qt.EditRole) -> bool:
        if role != Qt.EditRole or not index.isValid():
            return False
        ev = self._events[index.row()]
        col = index.column()
        try:
            if col == 0:
                ev.time = float(value)
            elif col == 1:
                ev.event = str(value)
            else:
                text = str(value).strip()
                ev.value = None if text == "" else (float(text) if _is_number(text) else text)
        except ValueError:
            return False
        self.dataChanged.emit(index, index)
        return True

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        return self.HEADERS[section] if orientation == Qt.Horizontal else section + 1


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False
