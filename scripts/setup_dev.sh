#!/usr/bin/env bash
pip install -e ".[dev]"
python -m spacy download en_core_web_trf
pre-commit install
