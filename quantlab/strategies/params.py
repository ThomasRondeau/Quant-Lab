"""
Parameter sheets: the typed description of every tunable value of a strategy.

Each strategy declares `PARAMS`, a tuple of `Param` (name, type, default, bounds, group, help text). The
sheet is the single reference for the CLI (`--param`, `--params-file`, `quantlab params`), the UI forms,
the optimizer ranges and the sessions. Constructors keep their explicit, typed arguments; a test checks
that sheet and constructor agree (names and defaults).

Presets are JSON files `{"strategy": NAME, "params": {...}}` (versioned in `presets/`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

Kind = Literal["int", "float", "bool", "choice", "tuple"]

#: display order of the groups (UI sections, `quantlab params`)
GROUPS = ("signal", "indicateurs", "modèle", "cible", "taille", "sorties")


class ParamError(ValueError):
    """Invalid parameter values (unknown key, wrong type, out of bounds, inconsistent values)."""


@dataclass(frozen=True)
class Param:
    name: str
    kind: Kind
    default: Any
    help: str = ""
    group: str = "signal"
    min: float | None = None
    max: float | None = None
    choices: tuple[Any, ...] = ()
    #: None is accepted ("none"); for numbers, 0 means "none" too
    optional: bool = False

    def __post_init__(self) -> None:
        if self.kind == "choice" and not self.choices:
            raise ValueError(f"{self.name}: a choice parameter needs choices.")
        if self.default is not None or not self.optional:
            object.__setattr__(self, "default", self.convert(self.default))

    @property
    def section(self) -> str:
        """Group without its qualifier (`modèle:direction` -> `modèle`)."""
        return self.group.split(":", 1)[0]

    def convert(self, value: Any) -> Any:
        """`value` (a Python value or text) as this parameter's type, within bounds; raises ParamError."""
        if isinstance(value, str) and value.strip().lower() in ("none", "null", ""):
            value = None
        if value is None:
            if self.optional:
                return None
            raise ParamError(f"{self.name}: a value is required.")
        try:
            converted = self._cast(value)
        except (TypeError, ValueError) as exc:
            raise ParamError(f"{self.name}: {value!r} is not a valid {self.kind}.") from exc
        self._check_bounds(converted)
        if self.optional and self.kind in ("int", "float") and converted == 0:
            return None  # 0 means "none" for an optional number
        return converted

    def _cast(self, value: Any) -> Any:
        if self.kind == "bool":
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in ("true", "1", "yes", "oui"):
                return True
            if text in ("false", "0", "no", "non"):
                return False
            raise ValueError(value)
        if self.kind == "int":
            if isinstance(value, bool):
                raise TypeError(value)
            number = float(value)
            if not number.is_integer():
                raise ValueError(value)
            return int(number)
        if self.kind == "float":
            if isinstance(value, bool):
                raise TypeError(value)
            return float(value)
        if self.kind == "tuple":
            items = value.strip("()[] ").split(",") if isinstance(value, str) else list(value)
            return tuple(int(float(item)) for item in items)
        # choice: match on the text, so "balanced" or "none" from the command line work
        for choice in self.choices:
            if value == choice or str(value) == str(choice):
                return choice
        raise ValueError(value)

    def _check_bounds(self, value: Any) -> None:
        if self.kind == "choice" and value not in self.choices:
            raise ParamError(f"{self.name}: {value!r} is not one of {list(self.choices)}.")
        numbers = value if self.kind == "tuple" else (value,) if self.kind in ("int", "float") else ()
        for number in numbers:
            if self.min is not None and number < self.min:
                raise ParamError(f"{self.name}: {value!r} is below the minimum {self.min:g}.")
            if self.max is not None and number > self.max:
                raise ParamError(f"{self.name}: {value!r} is above the maximum {self.max:g}.")

    def describe(self) -> dict[str, Any]:
        bounds = "" if self.min is None and self.max is None else f"[{_fmt(self.min)}, {_fmt(self.max)}]"
        if self.choices:
            bounds = " | ".join(map(str, self.choices))
        return {
            "group": self.group,
            "name": self.name,
            "type": self.kind,
            "default": self.default,
            "bounds": bounds,
            "help": self.help,
        }


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:g}"


def sheet_index(params: Iterable[Param]) -> dict[str, Param]:
    index: dict[str, Param] = {}
    for param in params:
        if param.name in index:
            raise ValueError(f"Parameter {param.name!r} is declared twice.")
        index[param.name] = param
    return index


def validate(params: Sequence[Param], values: Mapping[str, Any]) -> dict[str, Any]:
    """`values` converted to their types; unknown keys and invalid values raise ParamError (all at once)."""
    index = sheet_index(params)
    errors: list[str] = []
    converted: dict[str, Any] = {}
    for key, value in values.items():
        if key not in index:
            errors.append(f"unknown parameter {key!r}")
            continue
        try:
            converted[key] = index[key].convert(value)
        except ParamError as exc:
            errors.append(str(exc))
    if errors:
        raise ParamError("; ".join(e.rstrip(".") for e in errors) + f". Valid parameters: {', '.join(sorted(index))}.")
    return converted


def with_defaults(params: Sequence[Param], **new_defaults: Any) -> tuple[Param, ...]:
    """`params` with some defaults changed (a strategy overriding a base parameter's default)."""
    unknown = set(new_defaults) - {p.name for p in params}
    if unknown:
        raise ValueError(f"with_defaults: unknown parameters {sorted(unknown)}")
    return tuple(replace(p, default=new_defaults[p.name]) if p.name in new_defaults else p for p in params)


def defaults(params: Sequence[Param]) -> dict[str, Any]:
    return {param.name: param.default for param in params}


def diff(params: Sequence[Param], values: Mapping[str, Any]) -> dict[str, Any]:
    """The values that differ from the defaults."""
    index = sheet_index(params)
    return {k: v for k, v in values.items() if k not in index or index[k].default != v}


def param_range(param: Param, start: float, stop: float, step: float, limit: int = 1000) -> list[Any]:
    """`start`, `start + step`, ... up to `stop` (included) for a numeric parameter, each value checked."""
    if param.kind not in ("int", "float"):
        raise ParamError(f"{param.name}: a range needs a numeric parameter (it is {param.kind}).")
    if step <= 0 or stop < start:
        raise ParamError(f"{param.name}: invalid range {start}:{stop}:{step} (step > 0, stop >= start).")
    count = math.floor((stop - start) / step + 1e-9) + 1
    if count > limit:
        raise ParamError(f"{param.name}: {count} values in {start}:{stop}:{step} (at most {limit}).")
    raw = [start + i * step for i in range(count)]
    values = [round(v) if param.kind == "int" else round(v, 10) for v in raw]
    return [param.convert(v) for v in dict.fromkeys(values)]


# --- presets -----------------------------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def save_preset(path: str | Path, strategy: str, values: Mapping[str, Any]) -> None:
    payload = {"strategy": strategy, "params": {k: _jsonable(v) for k, v in values.items()}}
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_preset(path: str | Path) -> tuple[str, dict[str, Any]]:
    """(strategy name, raw parameter values) of a preset file; validate them with the strategy's sheet."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParamError(f"Cannot read the preset {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("strategy"), str):
        raise ParamError(f"{path}: a preset is {{'strategy': NAME, 'params': {{...}}}}.")
    values = payload.get("params", {})
    if not isinstance(values, dict):
        raise ParamError(f"{path}: 'params' must be an object.")
    return payload["strategy"], values
