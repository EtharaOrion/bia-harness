"""Deterministic extraction of an optimizer from a full training script.

A track-3 record ships as a complete trainer: it builds a model, opens data shards, and
initializes a process group at import time. Importing one directly is not possible in a
verifier. This module lifts just the optimizer and its transitive top-level dependencies
into a standalone source unit that can be imported and probed in isolation.

The procedure is pure syntax over the committed bytes. It parses the file, builds a
dependency graph over top-level definitions and assignments, takes the requested entry
symbol, closes over the names it references, and emits those definitions in their original
source order together with the file's imports. It never executes the source it reads while
deciding what to keep, so a hostile trainer cannot influence extraction.
"""
from __future__ import annotations

import ast
import hashlib


class _NameCollector(ast.NodeVisitor):
    def __init__(self):
        self.names = set()

    def visit_Name(self, node):
        self.names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        self.generic_visit(node)


def _referenced_names(node) -> set:
    c = _NameCollector()
    c.visit(node)
    return c.names


def _toplevel_units(tree):
    """Map top-level defined name to (index, node) preserving source order."""
    units = {}
    for i, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            units[node.name] = (i, node)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    units[tgt.id] = (i, node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            units[node.target.id] = (i, node)
    return units


def _import_lines(tree, source_lines):
    out = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.append("\n".join(source_lines[node.lineno - 1 : node.end_lineno]))
    return out


def extract_symbol(source: str, entry: str):
    """Return standalone source containing `entry` and its top-level closure.

    Raises KeyError when the entry symbol is not defined at top level, which is a hard
    failure rather than a silent empty extraction.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    units = _toplevel_units(tree)
    if entry not in units:
        raise KeyError(f"entry symbol not found at top level: {entry}")

    keep = {}
    frontier = [entry]
    seen = set()
    while frontier:
        name = frontier.pop()
        if name in seen or name not in units:
            continue
        seen.add(name)
        idx, node = units[name]
        keep[name] = (idx, node)
        for ref in _referenced_names(node):
            if ref in units and ref not in seen:
                frontier.append(ref)

    ordered = sorted(keep.values(), key=lambda t: t[0])
    parts = _import_lines(tree, lines)
    for _, node in ordered:
        parts.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
    return "\n\n".join(parts) + "\n"


def extraction_digest(source: str, entry: str) -> str:
    return hashlib.sha256(extract_symbol(source, entry).encode()).hexdigest()
