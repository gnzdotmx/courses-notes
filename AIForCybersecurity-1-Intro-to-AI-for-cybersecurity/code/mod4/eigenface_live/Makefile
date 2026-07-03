.PHONY: help install run test clean check-macos venv

PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

help:
	@echo "Eigenface Live (macOS only)"
	@echo ""
	@echo "  make install   Create venv and install dependencies"
	@echo "  make run       Launch the Qt application"
	@echo "  make test      Run unit tests"
	@echo "  make clean     Remove venv and Python cache files"

check-macos:
	@uname -s | grep -q Darwin || (echo "This project supports macOS only." && exit 1)

venv: check-macos
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run: check-macos install
	$(PY) main.py

test: check-macos install
	$(PY) -m unittest discover -s tests -v

clean:
	rm -rf $(VENV)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
