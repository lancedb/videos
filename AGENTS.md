# AGENTS.md

Guidance for AI agents working in this repository.

## Purpose

This repo stores code used to make videos that demonstrate how to use
[LanceDB](https://docs.lancedb.com). Each piece of code here exists to be shown,
explained, and run on camera — so clarity and correctness matter more than
cleverness.

## Environment

We use [uv](https://docs.astral.sh/uv/) to manage Python environments and
dependencies. Use `uv` commands (e.g. `uv add`, `uv run`, `uv sync`) rather than
`pip` or manual virtualenv management.

## Working principles

- **Keep it simple.** Prefer short, readable examples over abstractions. Code
  should be easy to follow for someone watching a video for the first time.
- **Make it runnable.** Every example should run end-to-end with minimal setup,
either using Marimo notebooks or regular Python scripts. Note any dependencies or setup steps clearly.
- **Explain intent.** Favor small, focused comments that explain the "why" when
  a step isn't obvious.
- **Ask before restructuring.** Since this code backs recorded videos, avoid
  large reorganizations without checking first.
