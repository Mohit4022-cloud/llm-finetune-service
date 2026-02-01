.PHONY: help install data train serve benchmark clean

help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies via Poetry"
	@echo "  make data        - Generate training data"
	@echo "  make train       - Fine-tune the model"
	@echo "  make serve       - Start FastAPI server"
	@echo "  make benchmark   - Run benchmark tests"
	@echo "  make clean       - Remove generated files"

install:
	poetry install

data:
	poetry run python scripts/data_gen.py

train:
	poetry run python scripts/train.py

serve:
	poetry run uvicorn src.api.app:app --host 0.0.0.0 --port 8000

benchmark:
	poetry run python scripts/benchmark.py

clean:
	rm -rf data/train.jsonl fine_tuned_adapters/ models/ REPORT.md
