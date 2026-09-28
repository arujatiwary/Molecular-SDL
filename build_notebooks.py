"""Build the .ipynb files from the cell scripts in notebooks/_src.

Each source file uses two delimiters:
    # %% [markdown]   -> markdown cell (lines after it start with '# ')
    # %%              -> code cell

Run:  python3 build_notebooks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

SRC = Path(__file__).parent / "notebooks" / "_src"
OUT = Path(__file__).parent / "notebooks"


def parse(path: Path):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if not text.strip():
            return
        if kind == "markdown":
            text = "\n".join(
                line[2:] if line.startswith("# ") else (line[1:] if line == "#" else line)
                for line in text.split("\n")
            )
            cells.append(nbf.v4.new_markdown_cell(text))
        else:
            cells.append(nbf.v4.new_code_cell(text))

    for line in path.read_text().split("\n"):
        if line.startswith("# %% [markdown]"):
            flush()
            kind, buf = "markdown", []
        elif line.startswith("# %%"):
            flush()
            kind, buf = "code", []
        else:
            buf.append(line)
    flush()
    return cells


def main():
    """Rebuild every notebook, or only those whose name contains an argument."""
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = sys.argv[1:]
    made = []
    for src in sorted(SRC.glob("*.py")):
        if wanted and not any(w in src.stem for w in wanted):
            continue
        nb = nbf.v4.new_notebook(cells=parse(src))
        nb.metadata = {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": sys.version.split()[0]},
        }
        dest = OUT / (src.stem + ".ipynb")
        nbf.write(nb, dest)
        made.append(dest.name)
    print("built:", ", ".join(made))


if __name__ == "__main__":
    main()
