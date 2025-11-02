from typing import List, Dict, Any, Optional
from qrooper.agents.llm_calls import QrooperLLM
from qrooper.prompts import (
    RECONNAISSANCE_COMPRESSION_SYSTEM_PROMPT,
    RECONNAISSANCE_COMPRESS_CONVERSATION_SYSTEM_PROMPT,
    RECONNAISSANCE_COMPRESS_ACCUMULATED_CONTEXT_SYSTEM_PROMPT
)

class ContextManagerAgent:
    """
    Context management agent that provides compression capabilities for:
    - LLM responses and tool usage in reconnaissance
    - Conversation histories for codebase analysis
    - Accumulated context compression to prevent bloat

    Optimized for codebase reconnaissance and architecture discovery workflows.
    """

    def __init__(self, desc: str = "ContextManager", model: str = "gemini-2.5-flash", reasoning_effort: str = "medium", temperature: float = 0.3):
        self.llm = QrooperLLM(desc=desc, model=model, reasoning_effort=reasoning_effort)
        self.temperature = temperature
        self.model = model
        self.reasoning_effort = reasoning_effort

        # Context tracking
        self.compressed_history = []
        self.key_insights = []
        self.exploration_priorities = []

    
    def compress_tool_interaction(self, llm_response: str, tool_use: str, tool_output: str) -> str:
        """Compress individual tool interactions to prevent context bloat."""
        # Limit tool output to prevent context overflow
        limited_tool_output = tool_output[:50000] if tool_output else ""

        prompt = RECONNAISSANCE_COMPRESSION_SYSTEM_PROMPT.format(
            llm_response=llm_response,
            tool_use=tool_use,
            limited_tool_output=limited_tool_output
        )

        try:
            response = self.llm.call(
                prompt_or_messages=prompt,
                model=self.model,
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort
            )

            # Handle different response formats
            if isinstance(response, dict):
                return response.get('content', str(response))
            return response
        except Exception as e:
            print(f"Error in compress_tool_interaction: {str(e)}")
            # Fallback compression
            return f"Reconnaissance action: {llm_response[:100]}... Tool: {tool_use}. Result: {tool_output[:200] if tool_output else 'No output detected'}..."

    def compress_conversation(self, conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Compress conversation history to maintain context efficiency."""
        # Convert conversation list to string format
        conversation_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])

        prompt = RECONNAISSANCE_COMPRESS_CONVERSATION_SYSTEM_PROMPT.format(conversation_str=conversation_str)

        try:
            response = self.llm.call(
                prompt_or_messages=prompt,
                model=self.model,
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort
            )

            # Handle different response formats
            if isinstance(response, dict):
                output = response.get('content', str(response))
            else:
                output = response

            output = "To reduce context size, here is a reconnaissance summary of previous conversation:\n" + output
            return [{"role": "user", "content": output}]
        except Exception as e:
            print(f"Error in compress_conversation: {str(e)}")
            # Fallback summary
            fallback_summary = f"Reconnaissance conversation summary: {len(conversation)} messages covering codebase analysis and architecture discovery."
            return [{"role": "user", "content": fallback_summary}]

    def compress_accumulated_context(self, accumulated_findings: List[str],
                                    files_explored: int, directories_explored: int,
                                    current_iteration: int, max_iterations: int) -> Dict[str, Any]:
        """Compress accumulated reconnaissance context and provide strategic guidance."""

        findings_text = "\n".join(accumulated_findings[-20:]) if accumulated_findings else "No findings yet"

        prompt = RECONNAISSANCE_COMPRESS_ACCUMULATED_CONTEXT_SYSTEM_PROMPT.format(
            current_iteration=current_iteration,
            max_iterations=max_iterations,
            files_explored=files_explored,
            directories_explored=directories_explored,
            accumulated_findings=findings_text
        )

        try:
            response = self.llm.call(
                prompt_or_messages=prompt,
                model=self.model,
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort
            )

            # Handle different response formats
            if isinstance(response, dict):
                output = response.get('content', str(response))
            else:
                output = response

            return self._parse_compressed_context(output)
        except Exception as e:
            print(f"Error in compress_accumulated_context: {str(e)}")
            # Fallback compressed context
            return {
                "architecture_summary": f"Codebase analysis with {files_explored} files and {directories_explored} directories explored.",
                "key_insights": ["Reconnaissance in progress", "Context compression activated"],
                "information_gaps": ["Detailed architectural analysis pending"],
                "next_priorities": ["Continue systematic exploration"],
                "completion_assessment": min(current_iteration / max_iterations * 100, 50)
            }

    def _parse_compressed_context(self, response: str) -> Dict[str, Any]:
        """Parse the compressed context response into structured data."""
        # Simple parsing - in a real implementation, this would be more sophisticated
        lines = response.split('\n')

        architecture_summary = ""
        key_insights = []
        information_gaps = []
        next_priorities = []
        completion_assessment = 0

        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if "architecture summary" in line.lower() or "essential architecture" in line.lower():
                current_section = "architecture"
            elif "critical insights" in line.lower() or "key findings" in line.lower():
                current_section = "insights"
            elif "information gaps" in line.lower() or "missing information" in line.lower():
                current_section = "gaps"
            elif "strategic next steps" in line.lower() or "next priorities" in line.lower():
                current_section = "priorities"
            elif "completion assessment" in line.lower():
                # Extract percentage from completion assessment
                import re
                percentages = re.findall(r'(\d+)%', line)
                if percentages:
                    completion_assessment = int(percentages[0])
            elif line.startswith('•') or line.startswith('-'):
                content = line[1:].strip()
                if current_section == "insights":
                    key_insights.append(content)
                elif current_section == "gaps":
                    information_gaps.append(content)
                elif current_section == "priorities":
                    next_priorities.append(content)
            elif current_section == "architecture" and line:
                architecture_summary += line + " "

        return {
            "architecture_summary": architecture_summary.strip(),
            "key_insights": key_insights[:5],  # Limit to 5 insights
            "information_gaps": information_gaps[:3],  # Limit to 3 gaps
            "next_priorities": next_priorities[:3],  # Limit to 3 priorities
            "completion_assessment": completion_assessment
        }

    def get_context_stats(self, text: str) -> Dict[str, Any]:
        """Get statistics about context size and compression recommendations."""
        char_count = len(text)
        # Rough estimation: 1 token ≈ 4 characters
        estimated_tokens = char_count // 4

        # Provide recommendations based on size for reconnaissance workflows
        if estimated_tokens < 1000:
            recommendation = "Small context - suitable for detailed reconnaissance analysis without compression"
        elif estimated_tokens < 5000:
            recommendation = "Medium context - consider compression to focus on critical architectural insights"
        elif estimated_tokens < 20000:
            recommendation = "Large context - compression recommended to highlight key codebase insights"
        else:
            recommendation = "Very large context - compression required for effective reconnaissance analysis"

        return {
            "character_count": char_count,
            "estimated_tokens": estimated_tokens,
            "recommendation": recommendation,
            "should_compress": estimated_tokens > 8000,
            "reconnaissance_complexity": "high" if estimated_tokens > 15000 else "medium" if estimated_tokens > 5000 else "low"
        }

    def should_trigger_context_compression(self, current_iteration: int,
                                          accumulated_context_size: int) -> bool:
        """Determine if context compression should be triggered."""
        # Trigger compression based on multiple factors
        trigger_conditions = [
            current_iteration % 3 == 0,  # Every 3 iterations
            accumulated_context_size > 10000,  # Large context
            current_iteration > 8  # Later iterations
        ]

        return any(trigger_conditions)

    def maintain_strategic_context(self, compressed_context: Dict[str, Any]) -> None:
        """Maintain strategic context across compression cycles."""
        # Store key insights persistently
        if compressed_context.get("key_insights"):
            for insight in compressed_context["key_insights"]:
                if insight not in self.key_insights:
                    self.key_insights.append(insight)

        # Store exploration priorities
        if compressed_context.get("next_priorities"):
            self.exploration_priorities = compressed_context["next_priorities"]

        # Limit stored insights to prevent memory bloat
        if len(self.key_insights) > 20:
            self.key_insights = self.key_insights[-15:]  # Keep last 15 insights