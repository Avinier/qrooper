"""
DeciderAgent - Intelligent query analysis for determining analysis depth

Analyzes queries to determine the optimal number of analysis passes needed:
- 1 Pass: Simple questions that need reconnaissance only
- 2 Passes: Pattern/relationship questions needing recon + pattern
- 3 Passes: Complex questions needing recon + pattern + deep analysis
"""

import json
import asyncio
from typing import Optional
from dataclasses import dataclass

from .llm_calls import QrooperLLM
from ..prompts import DECIDER_AGENT_PROMPT


@dataclass
class DecisionResult:
    """Result from DeciderAgent analysis"""
    passes_required: str  # "one", "two", or "three"
    reasoning: str


class DeciderAgent:
    """
    Intelligent decider that analyzes queries and determines optimal analysis strategy.
    Runs once at the beginning to avoid redundant passes and improve efficiency.
    """

    def __init__(self, llm_provider: QrooperLLM):
        """
        Initialize DeciderAgent

        Args:
            llm_provider: LLM provider for making decisions
        """
        self.llm = llm_provider
        self.decider_prompt = DECIDER_AGENT_PROMPT

    def decide(self, query: str) -> DecisionResult:
        """
        Analyze query and determine optimal analysis strategy

        Args:
            query: User's query

        Returns:
            DecisionResult with analysis strategy
        """
        prompt = self.decider_prompt.format(query=query)

        try:
            response = self.llm.call(
                prompt_or_messages=prompt,
                temperature=0.2,  # Low temperature for consistent decisions
                max_tokens=500
            )

            # Extract JSON from response if it's wrapped in markdown
            json_str = response
            if "```json" in response:
                # Extract JSON from markdown code block
                start = response.find("```json") + 7
                end = response.find("```", start)
                if end != -1:
                    json_str = response[start:end].strip()
            elif "```" in response:
                # Extract from any code block
                start = response.find("```") + 3
                end = response.find("```", start)
                if end != -1:
                    json_str = response[start:end].strip()

            # Parse JSON response
            decision_data = json.loads(json_str)

            return DecisionResult(
                passes_required=decision_data.get('passes_required', 'three'),
                reasoning=decision_data.get('reasoning', 'No reasoning provided')
            )

        except Exception as e:
            print(f"Error in DeciderAgent: {e}")
            # Fallback to full analysis
            return DecisionResult(
                passes_required="three",
                reasoning="Error in decision, defaulting to full analysis"
            )


# Unit test - run with: python -m qrooper.agents.decider
if __name__ == "__main__":
    from qrooper.agents.llm_calls import QrooperLLM

    def test_decider():
        print("=" * 60)
        print("Testing DeciderAgent")
        print("=" * 60)

        # Test query
        query = "analyse n give me the techstack of this codebase"
        model = "glm-4.5-air"
        reasoning_effort = "medium"

        # Initialize LLM provider with model and reasoning_effort
        llm = QrooperLLM(model=model, reasoning_effort=reasoning_effort)

        # Initialize DeciderAgent
        decider = DeciderAgent(llm)

        print(f"\nQuery: {query}")
        print(f"Model: {model}")
        print(f"Reasoning Effort: {reasoning_effort}")
        print("-" * 60)

        try:
            # Make decision
            result = decider.decide(query)

            print(f"\nDecision Result:")
            print(f"Passes Required: {result.passes_required}")
            print(f"Reasoning: {result.reasoning}")

        except Exception as e:
            print(f"Error during test: {e}")

    # Run the test
    test_decider()