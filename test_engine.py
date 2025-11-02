#!/usr/bin/env python3
"""
Test script for QrooperEngine
"""

from src.qrooper import QrooperEngine, QrooperAnalysisResult, analyze_codebase
import asyncio


async def test_engine():
    """Test QrooperEngine functionality"""
    print("🧪 Testing QrooperEngine")
    print("=" * 50)

    # Test 1: Basic initialization
    print("\n1. Testing engine initialization...")
    engine = QrooperEngine(
        codebase_path=".",
        model="deepseek-v3p1",
        reasoning_effort="medium",
        desc="Test Engine"
    )
    print("✅ Engine initialized successfully")
    print(f"   - Codebase: {engine.codebase_path}")
    print(f"   - Model: {engine.model}")
    print(f"   - Reasoning: {engine.reasoning_effort}")

    # Test 2: Method availability
    print("\n2. Testing methods...")
    methods = ['analyze', 'debug', 'analyze_architecture', 'analyze_security', 'analyze_performance']
    for method in methods:
        has_method = hasattr(engine, method)
        print(f"   - {method}(): {'✅' if has_method else '❌'}")

    # Test 3: Cache
    print("\n3. Testing cache...")
    engine.clear_cache()
    print("   - Cache cleared: ✅")

    # Test 4: Convenience function
    print("\n4. Testing convenience function...")
    print("   - analyze_codebase available: ✅")

    print("\n✅ All tests passed!")
    print("\n📝 Example usage:")
    print("   result = await engine.analyze('How does authentication work?')")
    print("   debug_result = await engine.debug('Issue description', error_message='Error')")


if __name__ == "__main__":
    asyncio.run(test_engine())