# Enterprise-to-Casual LLM Fine-Tuning Service

An LLM service that fine-tunes a model to rewrite formal corporate emails into friendly Slack messages. Built with FastAPI, includes Redis caching, rate limiting, and performance benchmarking.

## Features

- Fine-tuned LLM using LoRA adapters for efficient training
- FastAPI REST API with automatic documentation
- Redis caching with in-memory fallback
- Rate limiting (5 requests/minute)
- Performance benchmarking with detailed metrics
- Hardware-adaptive (CPU/MPS/CUDA)

## Quick Start

```bash
# Install dependencies
make install

# Generate training data
make data

# Fine-tune the model
make train

# Start API server
make serve

# In another terminal, run benchmark
make benchmark

# View results
cat REPORT.md
```

## Project Structure

```
llm-finetune-service/
├── pyproject.toml          # Poetry configuration
├── README.md               # This file
├── Makefile                # Common commands
├── .gitignore              # Git ignore patterns
├── .env.example            # Environment template
├── data/                   # Training data
│   └── train.jsonl
├── scripts/                # Executable scripts
│   ├── data_gen.py        # Generate training data
│   ├── train.py           # Fine-tuning script
│   └── benchmark.py       # Performance benchmarking
├── src/                    # Source code
│   └── api/
│       └── app.py         # FastAPI application
├── fine_tuned_adapters/   # Model adapters output
└── REPORT.md              # Benchmark results
```

## API Endpoints

### POST /generate
Generate casual Slack message from formal email.

**Request:**
```json
{
  "text": "I am writing to inform you that the project has been delayed."
}
```

**Response:**
```json
{
  "generated_text": "Hey! Quick update: the project has been delayed.",
  "source": "model",
  "latency_ms": 234
}
```

### GET /health
Check API health status.

## Configuration

Copy `.env.example` to `.env` and adjust settings as needed.

## Requirements

- Python 3.12+
- Poetry
- Redis (optional, falls back to in-memory cache)
- 4GB+ RAM for training

## License

MIT
