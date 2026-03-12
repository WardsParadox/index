#!/bin/env python3
# encoding: utf-8

# Copyright 2022-2025 Elliot Jordan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""build_processors.py

Builds a processor index (processors.json) by statically parsing AutoPkg
processor .py files using Python's ast module (no imports required).

Indexes built-in processors from autopkg/autopkg and community shared
processors from all other repos in the AutoPkg org.
"""

import argparse
import ast
import json
import os
import subprocess
import sys
from glob import glob

# Add v1/ to path so we can import from build.py when PA_TOKEN is available
sys.path.insert(0, os.path.dirname(__file__))

__version__ = "0.0.1"

PROCESSORS_PATH = "v1/processors.json"

# Known processor base classes in autopkglib
PROCESSOR_BASE_CLASSES = {
    "Processor",
    "DmgMounter",
    "URLGetter",
    "URLDownloader",
    "URLTextSearcher",
    "CURLDownloader",
    "CURLTextSearcher",
    "FileFinder",
}

# Files/dirs to skip when scanning
SKIP_FILENAMES = {
    "__init__.py",
    "xattr.py",
    "conftest.py",
    "setup.py",
}
SKIP_DIRS = {
    "__pycache__",
    ".git",
    "munkirepolibs",
    "autopkgyaml",
    "github",
}
SKIP_PREFIXES = ("test_",)


def should_skip_file(filepath):
    """Return True if this file should be skipped."""
    filename = os.path.basename(filepath)
    if filename in SKIP_FILENAMES:
        return True
    if any(filename.startswith(p) for p in SKIP_PREFIXES):
        return True
    # Skip any file inside a skip dir
    parts = filepath.replace("\\", "/").split("/")
    if any(part in SKIP_DIRS for part in parts):
        return True
    return False


def get_module_docstring(tree):
    """Extract the module-level docstring from an AST tree."""
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        return tree.body[0].value.value.strip()
    return None


def resolve_module_level_constants(tree):
    """Walk top-level assignments and collect simple constant values.

    Returns a dict mapping name -> resolved value.
    Handles str, int, float, bool, None, dict, list constants.
    """
    constants = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    val = _eval_simple(node.value)
                    if val is not None or isinstance(node.value, ast.Constant):
                        constants[target.id] = val
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value is not None:
                val = _eval_simple(node.value)
                if val is not None or isinstance(node.value, ast.Constant):
                    constants[node.target.id] = val
    return constants


def _eval_simple(node):
    """Evaluate a simple AST node to a Python value. Returns None if not simple."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        result = {}
        for k, v in zip(node.keys, node.values):
            if k is None:
                continue
            key = _eval_simple(k)
            val = _eval_simple(v)
            if key is not None:
                result[key] = val
        return result
    if isinstance(node, ast.List):
        return [_eval_simple(elt) for elt in node.elts]
    return None


def eval_ast_node(node, module_constants, module_docstring=None):
    """Recursively evaluate an AST node to a Python value.

    Handles the patterns found in actual autopkglib processor files.
    """
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        name = node.id
        if name in ("True", "False", "None"):
            return {"True": True, "False": False, "None": None}[name]
        if name in module_constants:
            return module_constants[name]
        # Fall back to the module docstring for bare __doc__ references that
        # weren't overridden in the constants dict (shouldn't normally happen).
        if name == "__doc__":
            return module_docstring
        return f"<{name}>"

    if isinstance(node, ast.Attribute):
        return "<dynamic>"

    if isinstance(node, ast.JoinedStr):
        # f-string
        return "<dynamic>"

    if isinstance(node, ast.Call):
        return "<dynamic>"

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = eval_ast_node(node.left, module_constants, module_docstring)
        right = eval_ast_node(node.right, module_constants, module_docstring)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        return "<dynamic>"

    if isinstance(node, ast.Dict):
        result = {}
        for key_node, val_node in zip(node.keys, node.values):
            if key_node is None:
                # **spread: val_node is the dict being spread
                spread = eval_ast_node(val_node, module_constants, module_docstring)
                if isinstance(spread, dict):
                    result.update(spread)
                elif isinstance(val_node, ast.Name) and val_node.id in module_constants:
                    spread_val = module_constants[val_node.id]
                    if isinstance(spread_val, dict):
                        result.update(spread_val)
            else:
                key = eval_ast_node(key_node, module_constants, module_docstring)
                val = eval_ast_node(val_node, module_constants, module_docstring)
                if key is not None:
                    result[key] = val
        return result

    if isinstance(node, ast.List):
        return [eval_ast_node(elt, module_constants, module_docstring) for elt in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(eval_ast_node(elt, module_constants, module_docstring) for elt in node.elts)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        val = eval_ast_node(node.operand, module_constants, module_docstring)
        if isinstance(val, bool):
            return not val
        return "<dynamic>"

    return "<dynamic>"


def resolve_class_level_constants(class_node, module_constants, module_docstring):
    """Collect simple class-level attribute assignments (e.g. nuspec_variables = {...}).

    These are needed to resolve **spread references inside input_variables dicts.
    Returns a dict that can be merged into module_constants for dict evaluation.
    """
    constants = {}
    for node in ast.iter_child_nodes(class_node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    val = eval_ast_node(node.value, module_constants, module_docstring)
                    constants[target.id] = val
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value is not None:
                val = eval_ast_node(node.value, module_constants, module_docstring)
                constants[node.target.id] = val
    return constants


def get_class_docstring(class_node):
    """Extract the docstring from a class definition node, or None."""
    if (
        class_node.body
        and isinstance(class_node.body[0], ast.Expr)
        and isinstance(class_node.body[0].value, ast.Constant)
        and isinstance(class_node.body[0].value.value, str)
    ):
        return class_node.body[0].value.value.strip()
    return None


def get_class_attr(class_node, attr_name, module_constants, module_docstring):
    """Extract and evaluate a class-level attribute by name.

    Handles both ast.Assign and ast.AnnAssign (type-annotated) assignments.
    Merges class-level constants into the lookup context so **spread references
    to other class attributes (e.g. **nuspec_variables) resolve correctly.

    Crucially, `__doc__` in the combined constants is set to the CLASS docstring
    (not the module docstring) because in Python a class body's `__doc__` is the
    class docstring — this is why `description = __doc__` yields the class
    description at runtime, not the module-level "See docstring for X" string.

    Returns None if the attribute is not found.
    """
    class_docstring = get_class_docstring(class_node)

    # Build combined constants: module-level first, then class-level attrs,
    # with __doc__ pointing at the CLASS docstring for correct `description = __doc__` resolution.
    class_constants = resolve_class_level_constants(
        class_node, module_constants, module_docstring
    )
    combined = {**module_constants, **class_constants}
    if class_docstring is not None:
        combined["__doc__"] = class_docstring

    for node in ast.iter_child_nodes(class_node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == attr_name:
                    return eval_ast_node(node.value, combined, module_docstring)
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == attr_name
                and node.value is not None
            ):
                return eval_ast_node(node.value, combined, module_docstring)
    return None


def get_base_class_names(class_node):
    """Return a list of base class name strings for a class definition node."""
    names = []
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def is_processor_class(class_node, module_constants):
    """Return True if this class looks like an AutoPkg processor."""
    base_names = get_base_class_names(class_node)
    # Inherits from a known processor base
    if any(b in PROCESSOR_BASE_CLASSES for b in base_names):
        return True
    # Defines input_variables as a class attribute (community processors)
    for node in ast.iter_child_nodes(class_node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "input_variables":
                    return True
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == "input_variables"
            ):
                return True
    return False


def extract_processor_info(filepath, source_root, repo, source_type):
    """Parse a .py file and extract processor class information.

    Args:
        filepath: Absolute path to the .py file.
        source_root: Root directory to compute relative paths from.
        repo: Full repo name (e.g., "autopkg/autopkg").
        source_type: "builtin" or "community".

    Returns:
        List of processor info dicts.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except OSError as e:
        print(f"  Warning: Could not read {filepath}: {e}")
        return []

    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        print(f"  Warning: Syntax error in {filepath}: {e}")
        return []

    module_docstring = get_module_docstring(tree)
    module_constants = resolve_module_level_constants(tree)

    # Add __doc__ to module_constants so class attrs can reference it
    if module_docstring:
        module_constants["__doc__"] = module_docstring

    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not is_processor_class(node, module_constants):
            continue

        class_name = node.name

        # Get description. `get_class_attr` resolves `description = __doc__` to the
        # class docstring (matching Python runtime behaviour), so this handles both
        # explicit string descriptions and the common `description = __doc__` pattern.
        # Fall back to the class docstring directly if no description attribute exists.
        description = get_class_attr(
            node, "description", module_constants, module_docstring
        )
        if description is None:
            description = get_class_docstring(node)
        if isinstance(description, str):
            description = description.strip()

        # Get input_variables and output_variables
        input_variables = get_class_attr(
            node, "input_variables", module_constants, module_docstring
        )
        output_variables = get_class_attr(
            node, "output_variables", module_constants, module_docstring
        )

        # Normalize: empty dict -> empty dict, dynamic -> None
        if isinstance(input_variables, str):
            input_variables = None
        if isinstance(output_variables, str):
            output_variables = None

        # Get lifecycle
        lifecycle = get_class_attr(
            node, "lifecycle", module_constants, module_docstring
        )
        if isinstance(lifecycle, str):
            lifecycle = None

        # Get parent class name
        base_names = get_base_class_names(node)
        parent_class = base_names[0] if base_names else None

        # Compute relative path
        rel_path = os.path.relpath(filepath, source_root).replace("\\", "/")

        # Build the processor key
        if source_type == "builtin":
            key = class_name
        else:
            key = f"{repo}/{class_name}"

        processor_info = {
            "name": class_name,
            "description": description,
            "source": source_type,
            "repo": repo,
            "path": rel_path,
            "parent_class": parent_class,
            "lifecycle": lifecycle if isinstance(lifecycle, dict) else None,
            "input_variables": input_variables,
            "output_variables": output_variables,
        }

        results.append((key, processor_info))

    return results


def scan_builtin_processors(autopkg_dir):
    """Scan built-in processors in autopkg/autopkg repo.

    Args:
        autopkg_dir: Path to the autopkg/autopkg checkout.

    Returns:
        List of (key, processor_info) tuples.
    """
    autopkglib_dir = os.path.join(autopkg_dir, "Code", "autopkglib")
    if not os.path.isdir(autopkglib_dir):
        print(f"  Warning: autopkglib dir not found at {autopkglib_dir}")
        return []

    results = []
    for py_file in sorted(glob(os.path.join(autopkglib_dir, "*.py"))):
        if should_skip_file(py_file):
            continue
        entries = extract_processor_info(
            py_file,
            source_root=autopkg_dir,
            repo="autopkg/autopkg",
            source_type="builtin",
        )
        results.extend(entries)

    return results


def scan_community_processors(repos_dir, repos):
    """Scan community processor .py files in all cloned org repos.

    Args:
        repos_dir: Path to the repos/ directory.
        repos: List of repo dicts from the GitHub API.

    Returns:
        List of (key, processor_info) tuples.
    """
    results = []
    for repo in repos:
        repo_name = repo["full_name"]
        repo_dir = os.path.join(repos_dir, repo_name)
        if not os.path.isdir(repo_dir):
            continue

        py_files = sorted(glob(os.path.join(repo_dir, "**", "*.py"), recursive=True))
        for py_file in py_files:
            if should_skip_file(py_file):
                continue
            entries = extract_processor_info(
                py_file,
                source_root=repo_dir,
                repo=repo_name,
                source_type="community",
            )
            results.extend(entries)

    return results


def get_repos_from_api():
    """Get repos from GitHub API using build.py's get_all_repos."""
    try:
        from build import clone_all_repos, get_all_repos
    except ImportError as e:
        print(f"  Warning: Could not import build.py: {e}")
        return None, None
    if not os.environ.get("PA_TOKEN"):
        return None, None
    try:
        repos = get_all_repos()
        return repos, clone_all_repos
    except Exception as e:
        print(f"  Warning: GitHub API call failed: {e}")
        return None, None


def discover_local_repos(repos_dir):
    """Discover repos already cloned in repos_dir.

    Returns a list of minimal repo dicts with just full_name.
    """
    repos = []
    if not os.path.isdir(repos_dir):
        return repos
    for org in os.listdir(repos_dir):
        org_dir = os.path.join(repos_dir, org)
        if not os.path.isdir(org_dir):
            continue
        for repo_name in os.listdir(org_dir):
            full_name = f"{org}/{repo_name}"
            # Skip the autopkg/autopkg repo -- it's handled separately as builtin
            if full_name == "autopkg/autopkg":
                continue
            repos.append({"full_name": full_name})
    return repos


def clone_autopkg_repo(autopkg_dir):
    """Clone autopkg/autopkg with --depth=1 if not already present."""
    if os.path.isdir(autopkg_dir):
        print(f"autopkg/autopkg already present at {autopkg_dir}, skipping clone.")
        return
    print("Cloning autopkg/autopkg...")
    os.makedirs(os.path.dirname(autopkg_dir), exist_ok=True)
    clone_cmd = [
        "git",
        "clone",
        "--depth=1",
        "https://github.com/autopkg/autopkg.git",
        autopkg_dir,
    ]
    subprocess.run(clone_cmd, check=True)


def build_processor_index(builtin_entries, community_entries):
    """Build the processors.json index structure."""
    processors = {}
    seen_keys = {}

    # Built-ins first
    for key, info in builtin_entries:
        if key in processors:
            print(f"  Warning: Duplicate builtin processor key '{key}', skipping.")
            continue
        processors[key] = info

    # Community
    for key, info in community_entries:
        if key in processors:
            # Deduplicate by appending a counter
            base_key = key
            counter = 2
            while key in processors:
                key = f"{base_key}_{counter}"
                counter += 1
            seen_keys[base_key] = key
            print(f"  Note: Duplicate community key, renamed to '{key}'")
        processors[key] = info

    total_builtin = sum(1 for v in processors.values() if v["source"] == "builtin")
    total_community = sum(1 for v in processors.values() if v["source"] == "community")

    return {
        "metadata": {
            "total_builtin": total_builtin,
            "total_community": total_community,
            "total": total_builtin + total_community,
        },
        "processors": processors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build AutoPkg processor index (processors.json)"
    )
    parser.add_argument(
        "--autopkg-dir",
        default=None,
        help=(
            "Path to a local autopkg/autopkg checkout. "
            "If not provided, repos/autopkg/autopkg will be cloned."
        ),
    )
    args = parser.parse_args()

    # Determine autopkg dir
    if args.autopkg_dir:
        autopkg_dir = os.path.abspath(args.autopkg_dir)
        print(f"Using local autopkg dir: {autopkg_dir}")
    else:
        autopkg_dir = os.path.abspath("repos/autopkg/autopkg")
        clone_autopkg_repo(autopkg_dir)

    # Get and clone org repos: try GitHub API first, fall back to local discovery
    repos_dir = "repos"
    repos, clone_fn = get_repos_from_api()
    if repos is not None:
        print(f"Found {len(repos)} org repos via GitHub API.")
        print("Cloning org repos (if not already present)...")
        clone_fn(repos)
    else:
        print(
            "PA_TOKEN not set or unavailable -- skipping GitHub API. "
            "Discovering already-cloned repos in repos/..."
        )
        repos = discover_local_repos(repos_dir)
        print(f"Found {len(repos)} locally cloned repo(s).")

    # Scan built-in processors
    print("\nScanning built-in processors...")
    builtin_entries = scan_builtin_processors(autopkg_dir)
    print(f"Found {len(builtin_entries)} built-in processor(s).")

    # Scan community processors
    print("\nScanning community processors...")
    community_entries = scan_community_processors("repos", repos)
    print(f"Found {len(community_entries)} community processor(s).")

    # Build index
    index = build_processor_index(builtin_entries, community_entries)

    # Write output
    os.makedirs(os.path.dirname(PROCESSORS_PATH), exist_ok=True)
    with open(PROCESSORS_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print()
    print("PROCESSOR INDEX SUMMARY:")
    print(f"  Built-in processors:  {index['metadata']['total_builtin']}")
    print(f"  Community processors: {index['metadata']['total_community']}")
    print(f"  Total processors:     {index['metadata']['total']}")
    print(f"  Output: {PROCESSORS_PATH}")


if __name__ == "__main__":
    main()
