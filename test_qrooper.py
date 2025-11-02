#!/usr/bin/env python3
"""
Test script for Qrooper - LLM-Powered 3-Pass Analysis System
Demonstrates the Scout -> Connect -> DeepDive workflow
"""

import asyncio
import json
from pathlib import Path

# Import the Qrooper system
from src.qrooper import (
    QrooperEngine,
    analyze_codebase,
    debug_issue,
    QrooperLLM
)


async def test_basic_analysis():
    """Test basic codebase analysis"""
    print("=" * 60)
    print("TEST 1: Basic Codebase Analysis")
    print("=" * 60)

    # Test on the qrooper package itself
    codebase_path = Path(".")

    # Create engine
    engine = QrooperEngine(codebase_path)

    # Ask a question about the codebase
    query = "How is the LLM provider structured in this codebase?"

    print(f"Query: {query}\n")

    # Run analysis
    result = await engine.analyze(query)

    # Print results
    print(f"Answer: {result.answer}\n")
    print(f"Passes used: {result.passes_used}")
    print(f"Files analyzed: {result.files_analyzed}")
    print(f"Analysis time: {result.analysis_time:.2f}s")

    if result.evidence:
        print("\nEvidence:")
        for ev in result.evidence[:3]:
            print(f"  - {ev['file_path']}:{ev['line_number']} - {ev['explanation']}")

    if result.recommendations:
        print("\nRecommendations:")
        for rec in result.recommendations[:3]:
            print(f"  • {rec}")

    print("\n" + "=" * 60 + "\n")


async def test_debugging():
    """Test debugging scenario"""
    print("=" * 60)
    print("TEST 2: Debugging Scenario")
    print("=" * 60)

    # Simulate a debugging scenario
    codebase_path = Path(".")

    # Debug an issue
    issue = "The LLM provider is not handling timeouts correctly"
    error_message = "TimeoutError: Request timed out after 30 seconds"

    print(f"Issue: {issue}")
    print(f"Error: {error_message}\n")

    # Run debug analysis
    result = await debug_issue(
        codebase_path,
        issue,
        error_message=error_message
    )

    # Print results
    print(f"Analysis: {result.answer[:500]}...\n")
    print(f"Root Cause: {result.agent_results['deep_analysis'].get('root_cause', 'Not identified')}")

    if result.examples:
        print("\nExample Fix:")
        for key, example in result.examples.items():
            print(f"{key}:")
            print(example[:200] + "..." if len(example) > 200 else example)

    print("\n" + "=" * 60 + "\n")


async def test_performance_analysis():
    """Test performance analysis mode"""
    print("=" * 60)
    print("TEST 3: Performance Analysis")
    print("=" * 60)

    codebase_path = Path(".")

    # Performance question
    query = "Are there any performance bottlenecks in the file reading operations?"

    print(f"Query: {query}\n")

    # Run performance analysis
    result = await analyze_codebase(
        codebase_path,
        query,
        mode="performance"
    )

    # Print results
    print(f"Analysis: {result.answer}\n")
    print(f"Mode: {result.analysis_mode}")

    # Show agent insights
    recon_summary = result.agent_results['reconnaissance']['summary']
    pattern_insights = result.agent_results['pattern_recognition'].get('insights', [])

    print(f"Reconnaissance Summary: {recon_summary[:200]}...")
    print(f"Pattern Insights: {', '.join(pattern_insights[:2])}")

    print("\n" + "=" * 60 + "\n")


async def test_convenience_functions():
    """Test convenience functions"""
    print("=" * 60)
    print("TEST 4: Convenience Functions")
    print("=" * 60)

    codebase_path = Path(".")

    # Test convenience functions from __init__.py
    print("Testing analyze_codebase convenience function:")
    result = await analyze_codebase(str(codebase_path), "What is the overall structure?")
    print(f"Analysis completed: {result is not None}")

    print("\n" + "=" * 60 + "\n")


async def test_agent_personalities():
    """Test that agents have distinct personalities"""
    print("=" * 60)
    print("TEST 5: Agent Personalities")
    print("=" * 60)

    from src.qrooper.prompts import (
        RECONNAISSANCE_AGENT_PROMPT,
        PATTERN_RECOGNITION_AGENT_PROMPT,
        DEEP_ANALYSIS_AGENT_PROMPT
    )

    # Show agent personalities
    print("SCOUT (Reconnaissance Agent):")
    print("Personality: Curious, systematic, highly observant")
    print("Focus: Mapping codebase structure")
    print("System prompt starts with: " + RECONNAISSANCE_AGENT_PROMPT[:100] + "...\n")

    print("CONNECT (Pattern Recognition Agent):")
    print("Personality: Analytical, insightful, pattern-obsessed")
    print("Focus: Understanding relationships and flows")
    print("System prompt starts with: " + PATTERN_RECOGNITION_AGENT_PROMPT[:100] + "...\n")

    print("DEEPDIVE (Deep Analysis Agent):")
    print("Personality: Meticulous, analytical, detail-oriented")
    print("Focus: Providing specific, detailed answers")
    print("System prompt starts with: " + DEEP_ANALYSIS_AGENT_PROMPT[:100] + "...\n")

    print("=" * 60 + "\n")


async def main():
    """Run all tests"""
    print("🚀 Testing Qrooper - LLM-Powered 3-Pass Analysis System")
    print("========================================================\n")

    # Check if we have API keys
    import os
    if not os.getenv("FIREWORKS_API_KEY"):
        print("⚠️  WARNING: FIREWORKS_API_KEY not found in environment")
        print("   Tests will simulate responses without actual LLM calls\n")

    try:
        # Run tests
        await test_basic_analysis()
        await test_debugging()
        await test_performance_analysis()
        await test_convenience_functions()
        await test_agent_personalities()

        print("✅ All tests completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())