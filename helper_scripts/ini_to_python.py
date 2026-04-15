#!/usr/bin/env python3
"""Convert an INI file to a Python module of @dataclasses.dataclass classes.

Usage:
    python ini_to_python.py <ini_path> [output_path]

Each INI section becomes a dataclass. Comments directly above a key become
that field's attribute docstring. Commented-out keys (e.g. ``#key = value``)
are included as commented-out field declarations.
"""
from __future__ import annotations

import argparse
import json
import keyword
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_name(name: str) -> str:
    """Return a safe Python identifier, appending ``_`` if ``name`` is a keyword."""
    return f"{name}_" if keyword.iskeyword(name) else name


def _section_to_classname(section: str) -> str:
    """Convert an INI section name to a valid PascalCase Python class name."""
    parts = re.split(r"[^a-zA-Z0-9]+", section)
    return "".join(part.title() for part in parts if part)


_MAX_LINE_LEN = 88


def _noqa(line: str) -> str:
    """Append ``  # noqa: E501`` if ``line`` exceeds the max line length."""
    if len(line) > _MAX_LINE_LEN:
        return f"{line}  # noqa: E501"
    return line


def _format_dict_literal(d: dict) -> str:
    """Format a parsed dict as a Python literal with 4-space indentation."""
    json_str = json.dumps(d, indent=4)
    # Replace JSON-only tokens with Python equivalents.
    json_str = re.sub(r"\btrue\b", "True", json_str)
    json_str = re.sub(r"\bfalse\b", "False", json_str)
    json_str = re.sub(r"\bnull\b", "None", json_str)
    return json_str


_INTERP_RE = re.compile(r"\$\{[^}]+\}")


def _infer_type_and_repr(value: str) -> tuple[str, str]:
    """Return ``(type_str, repr_str)`` inferred from an INI value string.

    Values that start with ``{`` and are valid JSON objects (after temporarily
    replacing ``${...}`` interpolations with placeholders) are typed as ``dict``
    and represented as a formatted Python dict literal with interpolations
    restored.  Multi-line values that are not dicts are typed as ``str``.
    Single-line values are checked for bool, int, float, then fall back to str.
    """
    stripped = value.strip()

    if not stripped:
        return "str", '""'

    # Dict detection: replace ${...} interpolations with temporary JSON-safe
    # placeholder strings, attempt JSON parsing, then restore originals.
    if stripped.startswith("{"):
        placeholder_map: dict[str, str] = {}
        counter = 0

        def _replace_interp(m: re.Match) -> str:
            nonlocal counter
            key = f"__INTERP_{counter}__"
            placeholder_map[key] = m.group(0)
            counter += 1
            # No surrounding quotes: ${...} always appears inside an existing
            # JSON string in the INI value, so the placeholder inherits them.
            return key

        substituted = _INTERP_RE.sub(_replace_interp, stripped)
        try:
            flat = re.sub(r"\s+", " ", substituted)
            parsed = json.loads(flat)
            if isinstance(parsed, dict):
                literal = _format_dict_literal(parsed)
                for key, original in placeholder_map.items():
                    literal = literal.replace(f'"{key}"', f'"{original}"')
                return "dict", literal
        except (json.JSONDecodeError, ValueError):
            pass

    # Multi-line: don't attempt further type inference.
    if "\n" in stripped:
        return "str", repr(stripped)

    if stripped.lower() == "true":
        return "bool", "True"
    if stripped.lower() == "false":
        return "bool", "False"

    try:
        int(stripped)
        return "int", stripped
    except ValueError:
        pass

    try:
        float(stripped)
        return "float", stripped
    except ValueError:
        pass

    return "str", repr(stripped)


def _format_field_docstring(comment_lines: list[str]) -> list[str]:
    """Return indented docstring lines for a field (4-space indent).

    ``comment_lines`` are the raw texts after stripping the leading ``#``
    from each INI comment line.
    """
    # Strip exactly one leading space (the space after '#') and trailing
    # whitespace. Preserves further indentation so formatted examples in
    # comments (e.g. indented dict values) remain readable in the docstring.
    texts = [(c[1:] if c.startswith(" ") else c).rstrip().replace("\\", "\\\\") for c in comment_lines]
    # Trim leading/trailing blank lines
    while texts and not texts[0]:
        texts.pop(0)
    while texts and not texts[-1]:
        texts.pop()
    if not texts:
        return []

    indent = "    "
    if len(texts) == 1:
        return [f'{indent}"""{texts[0]}"""']

    for i, t in enumerate(texts):
        if i == 0:
            result = [f'{indent}"""{t}']
        else:
            result.append(f"{indent}{t}" if t else "")
    result.append(f'{indent}"""')
    return result


def _format_class_docstring(comment_lines: list[str]) -> list[str]:
    """Return indented docstring lines for a class (4-space indent)."""
    return _format_field_docstring(comment_lines)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_BARE_KEY_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)$")
_ACTIVE_KEY_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*)")
_SECTION_RE = re.compile(r"^\[([^\]]+)\]")


def _parse(ini_path: Path) -> tuple[list[str], list[dict]]:
    """Parse an INI file.

    Returns
    -------
    pre_section_comments:
        Raw comment texts (after ``#``) that appear before the first section
        header.  These become the module docstring.

    sections:
        List of section dicts, each with:
        - ``name`` (str): the raw section name from the INI file
        - ``doc_lines`` (list[str]): comment texts for the class docstring
        - ``entries`` (list[dict]): fields and commented-out fields

        Each entry dict has:
        - ``type``: ``'field'`` or ``'commented_field'``
        - ``name`` (str): the INI key name
        - ``value`` (str | None): the value string, or None for bare keys
        - ``doc_lines`` (list[str]): comment texts above this entry
    """
    raw_lines = ini_path.read_text(encoding="utf-8").splitlines()
    n = len(raw_lines)

    pre_section_comments: list[str] = []
    sections: list[dict] = []
    current_section: dict | None = None
    pending_comments: list[str] = []  # comment texts (after #) awaiting a consumer

    i = 0
    while i < n:
        line = raw_lines[i]
        stripped = line.strip()

        # ------------------------------------------------------------------
        # Blank line
        # ------------------------------------------------------------------
        if not stripped:
            if current_section is None and pending_comments:
                # Still in the file header area – flush to pre-section block.
                pre_section_comments.extend(pending_comments)
                pre_section_comments.append("")
                pending_comments = []
            # Inside a section: leave pending_comments; they may belong to the
            # next key or the next section header.
            i += 1
            continue

        # ------------------------------------------------------------------
        # Section header  [name]
        # ------------------------------------------------------------------
        m = _SECTION_RE.match(stripped)
        if m:
            section_name = m.group(1)
            current_section = {
                "name": section_name,
                "doc_lines": pending_comments[:],
                "entries": [],
            }
            pending_comments = []
            sections.append(current_section)
            i += 1
            continue

        # ------------------------------------------------------------------
        # Comment line
        # ------------------------------------------------------------------
        if stripped.startswith("#"):
            comment_body = stripped[1:]  # keep the space for later .strip()

            # Accumulate all comment lines, including commented-out key/value
            # pairs, as plain comment text. They will flow into the docstring
            # of the next active field or the class.
            pending_comments.append(comment_body)
            i += 1
            continue

        # ------------------------------------------------------------------
        # Active key (with or without a value)
        # ------------------------------------------------------------------
        if current_section is not None:
            eq_idx = stripped.find("=")

            if eq_idx != -1:
                key = stripped[:eq_idx].strip()
                value = stripped[eq_idx + 1 :].strip()  # noqa: E203

                # Accumulate continuation lines (lines indented relative to key).
                while i + 1 < n and raw_lines[i + 1] and raw_lines[i + 1][0] in (" ", "\t"):
                    i += 1
                    value += "\n" + raw_lines[i].strip()

                current_section["entries"].append(
                    {
                        "type": "field",
                        "name": key,
                        "value": value,
                        "doc_lines": pending_comments[:],
                    }
                )
                pending_comments = []

            elif _BARE_KEY_RE.match(stripped):
                # Bare key with no value (e.g. ``cron_schedule``).
                current_section["entries"].append(
                    {
                        "type": "field",
                        "name": stripped,
                        "value": None,
                        "doc_lines": pending_comments[:],
                    }
                )
                pending_comments = []

        i += 1

    return pre_section_comments, sections


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def _generate(ini_path: Path, pre_section_comments: list[str], sections: list[dict]) -> str:
    """Generate the Python source from the parsed data."""
    out: list[str] = []

    # ---- Module docstring --------------------------------------------------
    # Strip trailing blank entries from the pre-section comment block.
    while pre_section_comments and not pre_section_comments[-1].strip():
        pre_section_comments.pop()

    pre_texts = [(c[1:] if c.startswith(" ") else c).rstrip() for c in pre_section_comments]

    if pre_texts:
        for i, t in enumerate(pre_texts):
            if i == 0:
                out.append(f'"""{t}')
            else:
                out.append(t)
    out.append("")
    out.append(f'Generated from: {ini_path}')
    out.append('"""')

    # ---- Imports -----------------------------------------------------------
    out.append("from __future__ import annotations")
    out.append("")
    out.append("import dataclasses")

    # ---- Classes -----------------------------------------------------------
    for section in sections:
        classname = _section_to_classname(section["name"])

        out.append("")
        out.append("")
        out.append("@dataclasses.dataclass")
        out.append(f"class {classname}:")

        # Class docstring (from comments that preceded the section header).
        class_doc = _format_class_docstring(section["doc_lines"])
        if class_doc:
            out.extend(class_doc)

        entries = section["entries"]
        if not entries:
            out.append("    pass")
            continue

        for entry in entries:
            out.append("")
            etype = entry["type"]
            name = entry["name"]
            value = entry["value"]
            doc_lines = entry["doc_lines"]

            if etype == "field":
                if value is None:
                    type_str = "str | None"
                    repr_str = "None"
                else:
                    type_str, repr_str = _infer_type_and_repr(value)

                safe = _safe_name(name)
                keyword_comment = f"  # INI key: {name}" if safe != name else ""

                if type_str == "dict":
                    dict_lines = repr_str.splitlines()
                    out.append(_noqa(f"    {safe}: dict = dataclasses.field({keyword_comment}"))
                    out.append(f"        default_factory=lambda: {dict_lines[0]}")
                    for dl in dict_lines[1:]:
                        out.append(f"        {dl}")
                    out.append("    )")
                else:
                    out.append(_noqa(f"    {safe}: {type_str} = {repr_str}{keyword_comment}"))

                field_doc = _format_field_docstring(doc_lines)
                if field_doc:
                    out.extend(field_doc)

    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert an INI file to a Python module of dataclasses."
    )
    parser.add_argument("ini_path", type=Path, help="Path to the source INI file.")
    parser.add_argument(
        "output_path",
        type=Path,
        nargs="?",
        help=(
            "Destination .py file. Defaults to the INI file with its extension "
            "replaced by .py (or .py appended when no .ini extension is present)."
        ),
    )
    args = parser.parse_args(argv)

    ini_path: Path = args.ini_path.resolve()
    if not ini_path.is_file():
        print(f"Error: {ini_path} is not a file.", file=sys.stderr)
        sys.exit(1)

    if args.output_path is not None:
        output_path: Path = args.output_path.resolve()
    else:
        if ini_path.suffix.lower() == ".ini":
            output_path = ini_path.with_suffix(".py")
        else:
            output_path = ini_path.with_name(ini_path.name + ".py")

    pre_section_comments, sections = _parse(ini_path)
    source = _generate(ini_path, pre_section_comments, sections)

    output_path.write_text(source, encoding="utf-8")
    print(f"Written: {output_path}")


if __name__ == "__main__":
    #main()
    main(["orthoseg/project_defaults.ini"])

