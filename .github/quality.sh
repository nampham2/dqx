#!/bin/bash

uv run ruff check
uv run mypy src tests examples
