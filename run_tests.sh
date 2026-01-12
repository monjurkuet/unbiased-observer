#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/knowledge_base
uv run python knowledge_base/tests/master_test.py
