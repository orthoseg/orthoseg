#!/usr/bin/env python3
"""Convert an INI file to a Sphinx RST file using ``.. confval::`` directives.

Usage:
    python ini_to_rst.py <ini_path> [output_path]

Each INI section becomes an RST section heading. Each parameter in the section
becomes a ``.. confval::`` directive with ``:type:``, ``:default:``, and the
comment above it as the description body.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# INI helpers
# ----------------------------------------------------------------------------

_BARE_KEY_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)$")
_ACTIVE_KEY_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*)")
_SECTION_RE = re.compile(r"^\[([^\]]+)\]")

_INTERP_RE = re.compile(r"\$\{[^}]+\}")


def _parse(ini_path: Path) -> tuple[list[str], list[dict]]:
    """Parse an INI file.

    Returns:
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
                # Still in the file header area, flush to pre-section block.
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
                value = stripped[eq_idx + 1 :].strip()

                # Accumulate continuation lines (lines indented relative to key).
                while (
                    i + 1 < n
                    and raw_lines[i + 1]
                    and raw_lines[i + 1][0] in (" ", "\t")
                ):
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


def _format_dict_literal(d: dict) -> str:
    """Format a parsed dict as a Python literal with 4-space indentation."""
    json_str = json.dumps(d, indent=4)
    # Replace JSON-only tokens with Python equivalents.
    json_str = re.sub(r"\btrue\b", "True", json_str)
    json_str = re.sub(r"\bfalse\b", "False", json_str)
    json_str = re.sub(r"\bnull\b", "None", json_str)
    return json_str


# ---------------------------------------------------------------------------
# RST helpers
# ---------------------------------------------------------------------------


def _section_heading(title: str, char: str = "-") -> list[str]:
    return [title, char * len(title)]


def _comment_to_rst(comment_lines: list[str]) -> list[str]:
    """Convert raw comment lines (text after ``#``) to RST body lines.

    Strips exactly one leading space after ``#`` (preserving further
    indentation) and right-strips trailing whitespace.
    """
    return [(c[1:] if c.startswith(" ") else c).rstrip() for c in comment_lines]


def _rst_default(type_str: str, repr_str: str) -> str:
    """Return a concise RST-formatted default value string."""
    if type_str == "dict":
        return "``{}`` (see description)" if repr_str == "{}" else "see description"
    if type_str in ("int", "float", "bool"):
        return f"``{repr_str}``"
    if repr_str == "None":
        return "``None``"
    # str: repr_str has surrounding quotes, strip them for display.
    inner = repr_str[1:-1] if repr_str and repr_str[0] in ("'", '"') else repr_str
    if not inner:
        return '``""``'
    return f"``{inner}``"


# ---------------------------------------------------------------------------
# RST generation
# ---------------------------------------------------------------------------


def _generate(
    ini_path: Path, pre_section_comments: list[str], sections: list[dict]
) -> str:
    out: list[str] = []

    # ---- Page title --------------------------------------------------------
    title = ini_path.stem.replace("_", " ").title()
    out.append(title)
    out.append("=" * len(title))
    out.append("")

    # Pre-section comments become an intro paragraph.
    intro = _comment_to_rst(pre_section_comments)
    # Trim leading/trailing blank lines.
    while intro and not intro[0].strip():
        intro.pop(0)
    while intro and not intro[-1].strip():
        intro.pop()
    if intro:
        out.extend(intro)
        out.append("")

    # ---- Sections ----------------------------------------------------------
    for section in sections:
        section_name = section["name"]
        sec_comments = _comment_to_rst(section["doc_lines"])
        while sec_comments and not sec_comments[0].strip():
            sec_comments.pop(0)
        while sec_comments and not sec_comments[-1].strip():
            sec_comments.pop()

        out.append("")
        out.extend(_section_heading(f"[{section_name}]"))
        out.append("")
        if sec_comments:
            out.extend(sec_comments)
            out.append("")

        for entry in section["entries"]:
            name = entry["name"]
            value = entry["value"]
            doc_lines = _comment_to_rst(entry["doc_lines"])

            # Trim leading/trailing blank lines from the description.
            while doc_lines and not doc_lines[0].strip():
                doc_lines.pop(0)
            while doc_lines and not doc_lines[-1].strip():
                doc_lines.pop()

            out.append(f".. confval:: {section_name}.{name}")

            if value is None:
                type_str = "str"
                default_rst = "``None``"
                repr_str = None
            else:
                type_str, repr_str = _infer_type_and_repr(value)
                default_rst = _rst_default(type_str, repr_str)

            out.append(f"   :type: ``{type_str}``")
            out.append(f"   :default: {default_rst}")
            out.append("")

            # For dict defaults with a non-trivial value, prepend a code block.
            if type_str == "dict" and repr_str and repr_str != "{}":
                code_lines = repr_str.splitlines()
                doc_lines = (
                    ["Default value:", "", ".. code-block:: python", ""]
                    + [f"   {line}" for line in code_lines]
                    + [""]
                    + doc_lines
                )

            if doc_lines:
                for line in doc_lines:
                    out.append(f"   {line}" if line.strip() else "")
                out.append("")
            else:
                out.append("")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the CLI.

    Args:
        argv (list[str] | None, optional): Command-line arguments. Defaults to None.
    """
    parser = argparse.ArgumentParser(
        description="Convert an INI file to a Sphinx RST file of confval directives."
    )
    parser.add_argument("ini_path", type=Path, help="Path to the source INI file.")
    parser.add_argument(
        "output_path",
        type=Path,
        nargs="?",
        help=(
            "Destination .rst file. Defaults to the INI file with its extension "
            "replaced by .rst (or .rst appended when no .ini extension is present)."
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
            output_path = ini_path.with_suffix(".rst")
        else:
            output_path = ini_path.with_name(ini_path.name + ".rst")

    pre_section_comments, sections = _parse(ini_path)
    source = _generate(ini_path, pre_section_comments, sections)

    output_path.write_text(source, encoding="utf-8")
    print(f"Written: {output_path}")


if __name__ == "__main__":
    # main()
    main(["orthoseg/project_defaults.ini", "docs/config_project.rst"])
