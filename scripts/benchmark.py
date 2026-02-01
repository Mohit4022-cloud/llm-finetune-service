#!/usr/bin/env python3
"""
Benchmark the Enterprise-to-Casual LLM API.
Tests latency, caching, and generates performance report.
"""

import requests
import time
import statistics
from typing import List, Dict

API_URL = "http://localhost:8000/generate"
HEALTH_URL = "http://localhost:8000/health"

TEST_INPUTS = [
    "I am writing to inform you that the project timeline has been extended by two weeks due to unforeseen technical complications. We apologize for any inconvenience this may cause.",
    "Please be advised that the meeting has been rescheduled to next Tuesday at 2:00 PM. Please confirm your attendance at your earliest convenience.",
    "We regret to inform you that your request cannot be processed at this time due to insufficient documentation. Please resubmit with the required materials.",
    "I would like to schedule a meeting to discuss the quarterly results and strategic planning for the next fiscal year. Please advise your availability.",
    "Thank you for your attention to this matter. Should you have any questions or concerns, please do not hesitate to reach out to our team."
]


def check_api_health() -> Dict:
    """Check if API is running and healthy."""
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error: API not accessible at {HEALTH_URL}")
        print(f"   {str(e)}")
        print("\n💡 Start the API first: make serve")
        exit(1)


def send_request(text: str) -> Dict:
    """Send a generation request to the API."""
    try:
        response = requests.post(
            API_URL,
            json={"text": text},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return {
            "generated_text": data["generated_text"],
            "source": data["source"],
            "latency_ms": data["latency_ms"],
            "success": True
        }
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            return {
                "generated_text": "",
                "source": "rate_limited",
                "latency_ms": 0,
                "success": False,
                "error": "Rate limit exceeded"
            }
        return {
            "generated_text": "",
            "source": "error",
            "latency_ms": 0,
            "success": False,
            "error": str(e)
        }
    except Exception as e:
        return {
            "generated_text": "",
            "source": "error",
            "latency_ms": 0,
            "success": False,
            "error": str(e)
        }


def run_benchmark() -> Dict:
    """Run benchmark tests."""
    results = {
        "requests": [],
        "cache_hits": 0,
        "cache_misses": 0,
        "latencies": [],
        "errors": 0
    }

    print("🚀 Starting benchmark...\n")
    print("📊 Test plan:")
    print("   - Phase 1: 5 unique requests (expect cache miss)")
    print("   - Phase 2: 5 repeated requests (expect cache hit)")
    print("   - Total: 10 requests\n")
    print("=" * 60)

    # Phase 1: 5 unique requests (cache miss expected)
    print("\n📤 PHASE 1: Unique requests (testing model)")
    print("-" * 60)

    for i, text in enumerate(TEST_INPUTS):
        print(f"\n[{i+1}/10] Sending unique request...")
        print(f"   Input: {text[:60]}...")

        response = send_request(text)
        results["requests"].append(response)

        if response["success"]:
            results["latencies"].append(response["latency_ms"])

            if response["source"] == "cache":
                results["cache_hits"] += 1
                print(f"   ✅ Response: {response['generated_text'][:60]}...")
                print(f"   ⚡ Cache HIT (unexpected): {response['latency_ms']}ms")
            else:
                results["cache_misses"] += 1
                print(f"   ✅ Response: {response['generated_text'][:60]}...")
                print(f"   🔄 Cache MISS: {response['latency_ms']}ms")
        else:
            results["errors"] += 1
            print(f"   ❌ Error: {response.get('error', 'Unknown error')}")

        time.sleep(0.5)  # Small delay between requests

    # Phase 2: Repeat same 5 requests (cache hit expected)
    print("\n\n📥 PHASE 2: Repeated requests (testing cache)")
    print("-" * 60)

    for i, text in enumerate(TEST_INPUTS):
        print(f"\n[{i+6}/10] Repeating request {i+1}...")
        print(f"   Input: {text[:60]}...")

        response = send_request(text)
        results["requests"].append(response)

        if response["success"]:
            results["latencies"].append(response["latency_ms"])

            if response["source"] == "cache":
                results["cache_hits"] += 1
                print(f"   ✅ Response: {response['generated_text'][:60]}...")
                print(f"   ⚡ Cache HIT: {response['latency_ms']}ms")
            else:
                results["cache_misses"] += 1
                print(f"   ✅ Response: {response['generated_text'][:60]}...")
                print(f"   🔄 Cache MISS (unexpected): {response['latency_ms']}ms")
        else:
            results["errors"] += 1
            print(f"   ❌ Error: {response.get('error', 'Unknown error')}")

        time.sleep(0.5)

    print("\n" + "=" * 60)

    return results


def calculate_metrics(results: Dict) -> Dict:
    """Calculate performance metrics."""

    latencies = results["latencies"]

    if not latencies:
        return {
            "total_requests": len(results["requests"]),
            "successful_requests": 0,
            "errors": results["errors"],
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_hit_rate": 0,
            "avg_latency_ms": 0,
            "p95_latency_ms": 0,
            "min_latency_ms": 0,
            "max_latency_ms": 0,
            "estimated_cost_per_1k": 0
        }

    successful = len(latencies)

    return {
        "total_requests": len(results["requests"]),
        "successful_requests": successful,
        "errors": results["errors"],
        "cache_hits": results["cache_hits"],
        "cache_misses": results["cache_misses"],
        "cache_hit_rate": (results["cache_hits"] / successful * 100) if successful > 0 else 0,
        "avg_latency_ms": statistics.mean(latencies),
        "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "estimated_cost_per_1k": calculate_cost(results)
    }


def calculate_cost(results: Dict) -> float:
    """
    Estimate cost per 1000 requests.

    Assumptions:
    - Base cost: $0.002 per request
    - Cache hit: 0.1x cost (10% of base)
    """
    base_cost = 0.002
    cache_hit_multiplier = 0.1

    successful = results["cache_hits"] + results["cache_misses"]

    if successful == 0:
        return 0

    total_cost = (
        results["cache_misses"] * base_cost +
        results["cache_hits"] * base_cost * cache_hit_multiplier
    )

    # Extrapolate to 1000 requests
    avg_cost_per_request = total_cost / successful
    return avg_cost_per_request * 1000


def generate_report(metrics: Dict, health: Dict) -> str:
    """Generate markdown benchmark report."""

    report = f"""# Enterprise-to-Casual LLM Service - Benchmark Report

## Test Configuration

- **API Endpoint:** {API_URL}
- **Test Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}
- **Test Strategy:** 10 sequential requests (5 unique + 5 repeated)
- **Model Type:** {health.get('model_type', 'unknown')}
- **Cache Type:** {health.get('cache_type', 'unknown')}

## Performance Metrics

### Request Statistics

- **Total Requests:** {metrics['total_requests']}
- **Successful:** {metrics['successful_requests']}
- **Errors:** {metrics['errors']}
- **Success Rate:** {metrics['successful_requests']/metrics['total_requests']*100:.1f}%

### Latency

- **Average Latency:** {metrics['avg_latency_ms']:.2f} ms
- **P95 Latency:** {metrics['p95_latency_ms']:.2f} ms
- **Min Latency:** {metrics['min_latency_ms']:.2f} ms
- **Max Latency:** {metrics['max_latency_ms']:.2f} ms

### Caching

- **Cache Hits:** {metrics['cache_hits']}
- **Cache Misses:** {metrics['cache_misses']}
- **Cache Hit Rate:** {metrics['cache_hit_rate']:.1f}%

### Cost Estimation

- **Estimated Cost per 1,000 Requests:** ${metrics['estimated_cost_per_1k']:.2f}
  - Assumes $0.002 base cost per request
  - Cache hits cost 10% of base (0.1x multiplier)

## Analysis

### Performance Assessment

"""

    # Performance analysis
    if metrics['p95_latency_ms'] < 500:
        report += "✅ **Excellent** - P95 latency is under 500ms\n"
    elif metrics['p95_latency_ms'] < 1000:
        report += "⚠️  **Good** - P95 latency is acceptable but could be optimized\n"
    else:
        report += "❌ **Needs Optimization** - P95 latency exceeds 1000ms\n"

    report += f"   - P95 latency: {metrics['p95_latency_ms']:.0f}ms\n"
    report += f"   - Average latency: {metrics['avg_latency_ms']:.0f}ms\n\n"

    # Caching analysis
    report += "### Caching Effectiveness\n\n"

    if metrics['cache_hit_rate'] >= 40:
        report += "✅ **Working Well** - Cache hit rate is healthy\n"
    elif metrics['cache_hit_rate'] >= 20:
        report += "⚠️  **Moderate** - Cache hit rate could be improved\n"
    else:
        report += "❌ **Low Hit Rate** - Cache may not be configured correctly\n"

    report += f"   - Cache hit rate: {metrics['cache_hit_rate']:.1f}%\n"
    report += f"   - {metrics['cache_hits']} hits vs {metrics['cache_misses']} misses\n\n"

    # Cost analysis
    report += "### Cost Efficiency\n\n"

    if metrics['estimated_cost_per_1k'] < 2:
        report += "✅ **Cost-Effective** - Estimated costs are low\n"
    elif metrics['estimated_cost_per_1k'] < 5:
        report += "⚠️  **Moderate Cost** - Acceptable for production use\n"
    else:
        report += "❌ **High Cost** - Consider optimization strategies\n"

    report += f"   - Estimated cost per 1k requests: ${metrics['estimated_cost_per_1k']:.2f}\n\n"

    # Recommendations
    report += """## Recommendations

### Latency Optimization

"""

    if metrics['p95_latency_ms'] > 1000:
        report += """- Consider model quantization (GGUF format) for faster CPU inference
- Evaluate using a smaller model architecture
- Implement request batching for multiple concurrent requests
"""
    else:
        report += "- Current latency is acceptable for production use\n"

    report += "\n### Caching Strategy\n\n"

    if metrics['cache_hit_rate'] >= 40:
        report += """- Cache is working effectively
- Maintain current TTL settings
- Monitor cache size and eviction patterns
"""
    else:
        report += """- Consider increasing cache TTL (currently set via REDIS_TTL env var)
- Implement cache warming for common queries
- Review cache key generation logic
"""

    report += f"\n### Scaling\n\n"

    throughput = 60 / (metrics['avg_latency_ms'] / 1000) if metrics['avg_latency_ms'] > 0 else 0

    report += f"""- Current throughput: ~{throughput:.1f} requests/minute (single worker)
- Add horizontal scaling if demand exceeds this threshold
- Consider async inference with queue system for high load
- Monitor rate limiting (currently 5 req/min per IP)

## Conclusion

"""

    if metrics['p95_latency_ms'] < 500 and metrics['cache_hit_rate'] >= 40:
        report += "The service is performing well and ready for production use. "
    elif metrics['errors'] > 0:
        report += "The service experienced errors during testing. Review logs and fix issues before production. "
    else:
        report += "The service is functional but could benefit from optimization. "

    report += f"Caching is {'working effectively' if metrics['cache_hit_rate'] >= 40 else 'available but underutilized'}. "

    if health.get('model_type') == 'mock':
        report += "\n\n⚠️  **Note:** Tests were run with mock model (rule-based). Train the model with `make train` for actual LLM inference.\n"

    return report


def main():
    """Main execution function."""

    print("\n" + "=" * 60)
    print("🎯 Enterprise-to-Casual LLM Service Benchmark")
    print("=" * 60 + "\n")

    # Check API health
    print("🔍 Checking API health...")
    health = check_api_health()
    print(f"✅ API is healthy")
    print(f"   Model: {health.get('model_type', 'unknown')}")
    print(f"   Cache: {health.get('cache_type', 'unknown')}")
    print()

    # Run benchmark
    results = run_benchmark()

    # Calculate metrics
    print("\n\n📊 Calculating metrics...")
    metrics = calculate_metrics(results)

    # Generate report
    print("📝 Generating report...")
    report = generate_report(metrics, health)

    # Save report
    with open("REPORT.md", "w") as f:
        f.write(report)

    print(f"✅ Report saved to REPORT.md\n")

    # Print summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"   Total Requests: {metrics['total_requests']}")
    print(f"   Successful: {metrics['successful_requests']}")
    print(f"   Errors: {metrics['errors']}")
    print(f"   Cache Hit Rate: {metrics['cache_hit_rate']:.1f}%")
    print(f"   Avg Latency: {metrics['avg_latency_ms']:.0f}ms")
    print(f"   P95 Latency: {metrics['p95_latency_ms']:.0f}ms")
    print(f"   Est. Cost/1k: ${metrics['estimated_cost_per_1k']:.2f}")
    print("=" * 60)
    print(f"\n✅ Benchmark complete! View full report: cat REPORT.md\n")


if __name__ == "__main__":
    main()
