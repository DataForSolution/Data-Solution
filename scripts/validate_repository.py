"""Fast, network-free structural checks for the curated portfolio."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        text=True,
    )
    return [ROOT / name for name in output.split("\0") if name]


def validate_python(files: list[Path]) -> int:
    python_files = [path for path in files if path.suffix == ".py"]
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return len(python_files)


def validate_notebooks(files: list[Path]) -> tuple[int, int]:
    notebooks = [path for path in files if path.suffix == ".ipynb"]
    code_cells = 0
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(notebook.get("cells"), list):
            raise ValueError(f"Notebook has no cells array: {path.relative_to(ROOT)}")
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", [])
            code = "".join(source) if isinstance(source, list) else source
            ast.parse(code, filename=f"{path}#cell-{index + 1}")
            code_cells += 1
    return len(notebooks), code_cells


def validate_markdown_links(files: list[Path]) -> int:
    markdown_files = [path for path in files if path.suffix.lower() == ".md"]
    link_pattern = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
    checked = 0
    failures: list[str] = []
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        for raw_target in link_pattern.findall(text):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            local_target = unquote(target.split("#", 1)[0])
            if not local_target:
                continue
            checked += 1
            if not (path.parent / local_target).resolve().exists():
                failures.append(f"{path.relative_to(ROOT)} -> {target}")
    if failures:
        raise ValueError("Broken local Markdown links:\n" + "\n".join(failures))
    return checked


def validate_sensitive_text(files: list[Path]) -> int:
    slash = chr(47)
    patterns = {
        "Google API key": re.compile("AI" + r"za[0-9A-Za-z_-]{35}"),
        "GitHub token": re.compile("gh" + r"[pousr]_[0-9A-Za-z]{20,}"),
        "AWS access key": re.compile("AK" + r"IA[0-9A-Z]{16}"),
        "private-key header": re.compile("BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY"),
        "Windows user path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.I),
        "macOS user path": re.compile(slash + "Users" + slash + r"[^/\s]+"),
        "Linux home path": re.compile(slash + "home" + slash + r"[^/\s]+"),
    }
    text_suffixes = {".md", ".py", ".toml", ".yml", ".yaml", ".json", ".txt"}
    scanned = 0
    failures: list[str] = []
    for path in files:
        if path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        scanned += 1
        for label, pattern in patterns.items():
            if pattern.search(text):
                failures.append(f"{path.relative_to(ROOT)}: {label}")
    if failures:
        raise ValueError("Sensitive or local-path patterns found:\n" + "\n".join(failures))
    return scanned


def main() -> None:
    files = tracked_files()
    python_count = validate_python(files)
    notebook_count, code_cell_count = validate_notebooks(files)
    link_count = validate_markdown_links(files)
    text_count = validate_sensitive_text(files)
    print(
        "Structural validation passed: "
        f"{python_count} Python files, {notebook_count} notebooks/"
        f"{code_cell_count} code cells, {link_count} local Markdown links, "
        f"{text_count} text files scanned."
    )


if __name__ == "__main__":
    main()
