# EquiBirth AI - reproducible entry points.
# Defaults to the POSIX venv layout. On Windows, override: make test PY=./.venv/Scripts/python.exe

PY ?= ./.venv/bin/python

.PHONY: install test lint typecheck clean presentation presentation-check

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

typecheck:
	$(PY) -m mypy src

# Rebuild the final presentation from the canonical reports/ artifacts, then audit it.
# Needs python-pptx, matplotlib, and pillow (see presentation/README.md).
presentation:
	$(PY) scripts/final_presentation_figures.py
	$(PY) scripts/final_presentation_build.py

# Layout audit (overflow, collisions), a claim-by-claim check against reports/, and a
# word-count check that the speaker script still fits the 10-minute slot.
presentation-check:
	$(PY) scripts/final_presentation_preview.py
	$(PY) scripts/final_presentation_verify.py
	$(PY) scripts/final_presentation_script_check.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__
