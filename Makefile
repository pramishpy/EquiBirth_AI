# EquiBirth AI - reproducible entry points.
# Defaults to the POSIX venv layout. On Windows, override: make test PY=./.venv/Scripts/python.exe

PY ?= ./.venv/bin/python

.PHONY: install test lint typecheck clean

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

typecheck:
	$(PY) -m mypy src

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__
