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
import sys
from pathlib import Path

# Re-use the parser and type inferrer from ini_to_python so both scripts
# stay in sync automatically.
sys.path.insert(0, str(Path(__file__).parent))
from ini_to_python import _infer_type_and_repr, _parse  # noqa: E402


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
        return "``{}`` (see description)"
    if type_str in ("int", "float", "bool"):
        return f"``{repr_str}``"
    if repr_str == "None":
        return "``None``"
    # str: repr_str has surrounding quotes – strip them for display.
    inner = repr_str[1:-1] if repr_str and repr_str[0] in ("'", '"') else repr_str
    if not inner:
        return '``""``'
    return f"``{inner}``"


# ---------------------------------------------------------------------------
# RST generation
# ---------------------------------------------------------------------------

def _generate(ini_path: Path, pre_section_comments: list[str], sections: list[dict]) -> str:
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
            else:
                type_str, repr_str = _infer_type_and_repr(value)
                default_rst = _rst_default(type_str, repr_str)

            out.append(f"   :type: ``{type_str}``")
            out.append(f"   :default: {default_rst}")
            out.append("")

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
    main()
