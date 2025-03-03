#!/bin/bash

mkdir -p .cov && rm -f .cov/*
COVERAGE_FILE=.cov/coverage_dqx uv run --no-sync pytest --cov-report= --cov=dqx tests
uv run coverage combine .cov/* && rm -fr .cov
uv run coverage report -m --skip-covered

rm -fr .cov/ .coverage
