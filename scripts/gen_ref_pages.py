"""Generate API reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files


nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "src"

EXCLUDE_DIRS = {"gunpowder", "swc_morphology", "neurotorch", "notebooks", "deprecated", "reconnect_stack_navis.py"}

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)
    
    
    if EXCLUDE_DIRS & set(path.parts):
        continue

    if parts[-1] == "__init__":
        # Package __init__ becomes the package index page.
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    if not parts:
        continue

    # Keep Python spelling exactly as-is in the generated nav.
    nav[parts] = doc_path.as_posix()

    identifier = ".".join(parts)
    title = parts[-1]

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        # Important for package/index pages:
        # prevents MkDocs from converting e.g.
        # ac_segmentation -> "Ac segmentation".
        print("---", file=fd)
        print(f"title: {title}", file=fd)
        print("---", file=fd)
        print(file=fd)

        print(f"::: {identifier}", file=fd)

    mkdocs_gen_files.set_edit_path(
        full_doc_path,
        path.relative_to(root),
    )

summary = list(nav.build_literate_nav())

print("\nGENERATED API NAV:")
print("".join(summary))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(summary)

