"""Generate API reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files


def path_is_excluded(path, excluded_paths=None):
    excluded_paths = excluded_paths or tuple()
    relative = path.relative_to(src)

    return any(
        relative.match(pattern)
        for pattern in excluded_paths
    )


def module_is_excluded(parts, skip_private_modules=False):
    identifier = ".".join(parts)

    if any(
        identifier == excluded
        or identifier.startswith(excluded + ".")
        for excluded in excluded_modules
    ):
        return True

    if skip_private_modules:
        if any(
            part.startswith("_")
            and part not in {"__init__", "__main__"}
            for part in parts
        ):
            return True

    return False


config = mkdocs_gen_files.config
api_config = config.get("extra", {}).get("api_reference", {})

source_dir = api_config.get("source_dir", "src")
output_dir = api_config.get("output_dir", "reference")

excluded_modules = set(
    api_config.get("excluded_modules", [])
)

excluded_paths = tuple(
    api_config.get("excluded_paths", [])
)

skip_private = api_config.get(
    "skip_private_modules",
    True,
)

nav = mkdocs_gen_files.Nav()

root = Path(config.config_file_path).parent.resolve()
src = (root / source_dir).resolve()

for path in sorted(src.rglob("*.py")):
    if path_is_excluded(path, excluded_paths):
        print(f"excluded {path}")
        continue
    print(f"did not exclude {path}")

    relative_path = path.relative_to(src)

    module_path = relative_path.with_suffix("")
    doc_path = relative_path.with_suffix(".md")

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        # Package __init__ becomes the package index page.
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")

    if not parts:
        continue

    if module_is_excluded(parts, skip_private):
        continue

    full_doc_path = Path(output_dir) / doc_path

    # Keep Python spelling exactly as-is in the generated nav.
    nav[parts] = doc_path.as_posix()

    identifier = ".".join(parts)

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        print(f"::: {identifier}", file=fd)

    mkdocs_gen_files.set_edit_path(
        full_doc_path,
        path.relative_to(root),
    )

summary_path = Path(output_dir) / "SUMMARY.md"
summary = list(nav.build_literate_nav())

print("\nGENERATED API NAV:")
print("".join(summary))

with mkdocs_gen_files.open(summary_path, "w") as nav_file:
    nav_file.writelines(summary)

