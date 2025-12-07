#!/bin/bash
export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "Running Traditional ML Baseline (SVM & Naive Bayes)..."
python src/compare_traditional.py