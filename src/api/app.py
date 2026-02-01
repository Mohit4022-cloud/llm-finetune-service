#!/usr/bin/env python3
"""
FastAPI service for Enterprise-to-Casual LLM inference.
Includes Redis caching, rate limiting, and graceful fallbacks.
"""

import os
import json
import hashlib
import time
from typing import Optional

import redis
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


# ============================================================================
# Cache Client
# ============================================================================

class CacheClient:
    """Redis cache with in-memory fallback."""

    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None

        # Try to connect to Redis
        try:
            self.redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=int(os.getenv("REDIS_DB", 0)),
                socket_connect_timeout=2,
                decode_responses=True
            )
            self.redis_client.ping()
            print("✅ Redis connected")
        except Exception as e:
            print(f"⚠️  Redis unavailable: {e}")
            print("   Using in-memory cache as fallback")
            self.redis_client = None

        # Fallback in-memory cache
        self.memory_cache = {}

    def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        if self.redis_client:
            try:
                return self.redis_client.get(key)
            except Exception:
                pass

        return self.memory_cache.get(key)

    def set(self, key: str, value: str, ttl: int = 3600):
        """Set value in cache with TTL."""
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, value)
                return
            except Exception:
                pass

        # Fallback to memory cache (no TTL in simple dict)
        self.memory_cache[key] = value


# ============================================================================
# Model Inference
# ============================================================================

class ModelInference:
    """Model loader with mock fallback for testing."""

    def __init__(self):
        adapter_path = os.getenv("ADAPTER_PATH", "./fine_tuned_adapters")
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")

        # Check if adapters exist
        if not os.path.exists(adapter_config_path):
            print("⚠️  No adapters found - using base model")
            self.mock_mode = True
            self.model = None
            self.tokenizer = None
            return

        # Load adapter config
        with open(adapter_config_path) as f:
            config = json.load(f)

        # Check if mock adapters
        if config.get("mock"):
            print("⚠️  Mock adapters detected - using rule-based generation")
            self.mock_mode = True
            self.model = None
            self.tokenizer = None
            return

        # Load real model with adapters
        print("📦 Loading fine-tuned model...")
        self.mock_mode = False

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            base_model = config["base_model_name_or_path"]

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load base model
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float32,
                device_map="auto"
            )

            # Load LoRA adapters
            self.model = PeftModel.from_pretrained(base, adapter_path)
            self.model.eval()

            print("✅ Model loaded with fine-tuned adapters")

        except Exception as e:
            print(f"⚠️  Failed to load model: {e}")
            print("   Falling back to mock mode")
            self.mock_mode = True
            self.model = None
            self.tokenizer = None

    def generate(self, text: str) -> str:
        """Generate casual message from formal text."""

        if self.mock_mode:
            return self._mock_generate(text)

        # Real model inference
        prompt = f"### Instruction:\nRewrite this corporate email into a casual, friendly Slack message.\n\n### Input:\n{text}\n\n### Response:\n"

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Move inputs to same device as model
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response after "### Response:\n"
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()

        return response

    def _mock_generate(self, text: str) -> str:
        """Simple rule-based transformation for testing."""

        casual = text

        # Replace formal phrases with casual equivalents
        replacements = {
            "I am writing to inform you that": "Hey! Quick update:",
            "I am writing to inform you": "Hey! Quick update:",
            "Please be advised that": "Just FYI -",
            "Please be advised": "Just FYI -",
            "I would like to schedule": "Can we schedule",
            "I would like to": "Can we",
            "This matter requires immediate attention": "Heads up - we need to handle this ASAP",
            "We regret to inform you that": "Small update - looks like",
            "We regret to inform you": "Small update - looks like",
            "Per our previous conversation": "Following up on our chat",
            "Per our conversation": "Following up on",
            "I would appreciate your input on": "Quick question -",
            "I would appreciate your input": "Quick question -",
            "I would like to provide feedback on": "Thoughts on",
            "I would like to provide feedback": "Thoughts on",
            "For the sake of clarity": "Just to clarify -",
            "We acknowledge receipt of": "Got it!",
            "Thank you for your attention to this matter": "Thanks!",
            "Sincerely": "Cheers",
            "Best regards": "Thanks",
            "Kind regards": "Thanks",
            "apologize for any inconvenience": "sorry for the delay! 😅",
        }

        for formal, friendly in replacements.items():
            casual = casual.replace(formal, friendly)

        # Truncate if too long
        if len(casual) > 250:
            casual = casual[:247] + "..."

        return casual


# ============================================================================
# FastAPI Application
# ============================================================================

# Initialize FastAPI
app = FastAPI(
    title="Enterprise-to-Casual LLM Service",
    description="Convert formal corporate emails into friendly Slack messages",
    version="0.1.0"
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize cache and model
cache = CacheClient()
model = ModelInference()


# ============================================================================
# Request/Response Models
# ============================================================================

class GenerateRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I am writing to inform you that the project has been delayed due to unforeseen circumstances."
            }
        }


class GenerateResponse(BaseModel):
    generated_text: str
    source: str  # "cache" or "model"
    latency_ms: int

    class Config:
        json_schema_extra = {
            "example": {
                "generated_text": "Hey! Quick update: the project has been delayed due to unforeseen circumstances.",
                "source": "model",
                "latency_ms": 234
            }
        }


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Enterprise-to-Casual LLM",
        "version": "0.1.0",
        "endpoints": {
            "generate": "/generate",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": not model.mock_mode,
        "model_type": "fine-tuned" if not model.mock_mode else "mock",
        "cache_available": cache.redis_client is not None,
        "cache_type": "redis" if cache.redis_client else "memory"
    }


@app.post("/generate", response_model=GenerateResponse)
@limiter.limit("5/minute")
async def generate(request: Request, data: GenerateRequest):
    """
    Generate casual Slack message from formal email.

    Rate limit: 5 requests per minute per IP.
    """

    start_time = time.time()

    # Validate input
    if not data.text or len(data.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(data.text) > 2000:
        raise HTTPException(status_code=400, detail="Text too long (max 2000 characters)")

    # Generate cache key
    cache_key = f"casual:{hashlib.md5(data.text.encode()).hexdigest()}"

    # Check cache
    cached = cache.get(cache_key)
    if cached:
        latency = int((time.time() - start_time) * 1000)
        return GenerateResponse(
            generated_text=cached,
            source="cache",
            latency_ms=latency
        )

    # Generate from model
    try:
        result = model.generate(data.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    # Cache result
    ttl = int(os.getenv("REDIS_TTL", 3600))
    cache.set(cache_key, result, ttl=ttl)

    latency = int((time.time() - start_time) * 1000)

    return GenerateResponse(
        generated_text=result,
        source="model",
        latency_ms=latency
    )


# ============================================================================
# Startup
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Print startup information."""
    print("\n" + "=" * 60)
    print("🚀 Enterprise-to-Casual LLM Service")
    print("=" * 60)
    print(f"   Model mode: {'Fine-tuned' if not model.mock_mode else 'Mock (rule-based)'}")
    print(f"   Cache: {'Redis' if cache.redis_client else 'In-memory'}")
    print(f"   Rate limit: 5 requests/minute")
    print("=" * 60)
    print("   Ready to serve requests!")
    print("   API docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")
