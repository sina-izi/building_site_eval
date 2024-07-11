# Makefile for common tasks

.PHONY: setup run clean

setup:
    python3 -m venv venv
    source venv/bin/activate && pip install -r requirements.txt

run:
    source venv/bin/activate && python 3d/surface_smoother.py

clean:
    rm -rf __pycache__
    rm -rf venv
