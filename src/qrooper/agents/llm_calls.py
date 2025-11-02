import os
import json
import requests
import logging
import traceback
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
import time
import threading

import google.genai as genai
from google.genai import types
from eva.schemas import FireworksTool, FireworksToolCallResponse, LlmMessage

class QrooperLLM:
    """
    Comprehensive LLM Helper class supporting multiple providers:
    - Google Gemini (basic, tool use, grounding)
    - Fireworks AI (DeepSeek, Qwen models)
    """

    def __init__(self, desc: str = "", model: str = "deepseek-v3p1", reasoning_effort: str = "medium", **_ignored_kwargs):
        # Store defaults for LLM calls
        self.desc = desc
        self.default_model = model
        self.default_reasoning_effort = reasoning_effort

        # Setup logging with DEBUG level
        logger_name = f"QrooperLLM.{desc or 'default'}"
        self.logger = logging.getLogger(logger_name)

        # Ensure logger outputs at DEBUG level
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('[%(asctime)s] [%(name)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

        # Configure root logger to show DEBUG messages
        logging.getLogger().setLevel(logging.DEBUG)

        self.logger.debug(f"🔧 Initialized QrooperLLM with model={model}, reasoning_effort={reasoning_effort}")

        # Initialize Gemini client
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if self.gemini_api_key:
            if genai:
                self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            else:
                self.gemini_client = None
                self.logger.warning("google-genai not installed")
        else:
            self.gemini_client = None
            self.logger.warning("GOOGLE_API_KEY not found in environment variables")

        # Fireworks AI configuration
        self.fireworks_api_key = os.getenv("FIREWORKS_API_KEY")
        self.fireworks_endpoint = "https://api.fireworks.ai/inference/v1/chat/completions"
        self.fireworks_headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.fireworks_api_key}"
        }

        # GLM/ZhipuAI configuration
        self.glm_api_key = os.getenv("GLM_API_KEY") or os.getenv("ZHIPU_API_KEY")
        # Use open.bigmodel.cn endpoint as per official docs
        self.glm_endpoint = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.glm_headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.glm_api_key}" if self.glm_api_key else ""
        }

        # Gemini model mapping
        self.gemini_models = {
            "gemini-2.5-flash": "gemini-2.5-flash",
            "gemini-2.5-pro": "gemini-2.5-pro"
        }

        # Fireworks model mapping
        self.fireworks_models = {
            "deepseek-v3p1": "accounts/fireworks/models/deepseek-v3p1",
            "deepseek-r1": "accounts/fireworks/models/deepseek-r1-0528",
            "qwen3-30b-a3b": "accounts/fireworks/models/qwen3-30b-a3b",
            "qwen2.5-72b-instruct": "accounts/fireworks/models/qwen2p5-72b-instruct"
        }

        # GLM model mapping - official model codes from docs.z.ai
        self.glm_models = {
            "glm-4.6": "glm-4.6",
            "glm-4.5": "glm-4.5",
            "glm-4.5-air": "glm-4.5-air",
            "glm-4.5-x": "glm-4.5-x",
            "glm-4.5-airx": "glm-4.5-airx",
            "glm-4.5-flash": "glm-4.5-flash",
            "glm-4-32b-0414-128k": "glm-4-32b-0414-128k",
            "glm-4.5v": "glm-4.5v"  # Vision model
        }


    #=======GOOGLE API CALLS=======
    def gemini_basic_call(self, prompt_or_messages, model: str = "gemini-2.5-flash",
                         system_prompt: Optional[str] = None, stream: bool = False,
                         on_token: Optional[Callable[[str], None]] = None,
                         on_reasoning: Optional[Callable[[str], None]] = None,
                         reasoning_effort: str = "medium",
                         **kwargs) -> str:
        """Basic Gemini API call for text generation with streaming support and thinking mode.

        Args:
            prompt_or_messages: Either a string prompt or a list of message dicts
            model: Model name (gemini-2.5-flash or gemini-2.5-pro)
            system_prompt: Optional system prompt to prepend
            stream: Enable streaming response
            on_token: Callback for each content token during streaming
            on_reasoning: Callback for reasoning/thinking tokens during streaming
            reasoning_effort: Reasoning effort level (none/low/medium/high) - maps to thinking_budget
            **kwargs: Additional parameters (unused for compatibility)

        Returns:
            String response from the model
        """
        if not self.gemini_client:
            raise ValueError("Gemini client not initialized. Check GOOGLE_API_KEY.")

        # Get the full model name from mapping
        full_model_name = self.gemini_models.get(model, model)

        # Handle different input types - similar to fw_basic_call
        contents = []
        if isinstance(prompt_or_messages, str):
            # Single prompt
            if system_prompt:
                contents = f"{system_prompt}\n\n{prompt_or_messages}"
            else:
                contents = prompt_or_messages
        elif isinstance(prompt_or_messages, list):
            # List of messages - convert to Gemini format
            # Gemini expects a simple string for basic calls or structured content
            # We'll concatenate messages into a single prompt
            combined_prompt = ""
            if system_prompt:
                combined_prompt += f"System: {system_prompt}\n\n"

            for msg in prompt_or_messages:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    # LlmMessage object
                    role = msg.role
                    content = msg.content
                elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    # Dict format
                    role = msg['role']
                    content = msg['content']
                else:
                    raise ValueError("Invalid message format. Expected LlmMessage objects or dicts with 'role' and 'content' keys.")

                if role == "system":
                    combined_prompt += f"System: {content}\n\n"
                elif role == "user":
                    combined_prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    combined_prompt += f"Assistant: {content}\n\n"
                else:
                    combined_prompt += f"{role}: {content}\n\n"

            contents = combined_prompt.strip()
        else:
            raise ValueError("Input must be either a string prompt or a list of messages.")

        # Map reasoning_effort to thinking_budget
        # -1 = dynamic (recommended), 0 = disabled, or specific token count
        thinking_budget_map = {
            "none": 0,        # No thinking/reasoning
            "low": 4096,      # Minimal thinking
            "medium": -1,     # Dynamic thinking (recommended)
            "high": 16384     # Extended thinking
        }
        thinking_budget = thinking_budget_map.get(reasoning_effort, -1)

        # Create config with thinking support
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=thinking_budget
            )
        )

        try:
            if stream:
                # Streaming response with thinking support
                response_stream = self.gemini_client.models.generate_content_stream(
                    model=full_model_name,
                    contents=contents,
                    config=config
                )

                full_text = []
                full_thoughts = []

                for chunk in response_stream:
                    if chunk.candidates and len(chunk.candidates) > 0:
                        candidate = chunk.candidates[0]
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                # Check if this is a thought (reasoning) part
                                if hasattr(part, 'thought') and part.thought:
                                    # This is reasoning/thinking content
                                    if hasattr(part, 'text') and part.text:
                                        full_thoughts.append(part.text)
                                        if on_reasoning:
                                            try:
                                                on_reasoning(part.text)
                                            except Exception:
                                                pass
                                # Regular content part
                                elif hasattr(part, 'text') and part.text:
                                    full_text.append(part.text)
                                    if on_token:
                                        try:
                                            on_token(part.text)
                                        except Exception:
                                            pass

                return ''.join(full_text)
            else:
                # Non-streaming response with thinking support
                response = self.gemini_client.models.generate_content(
                    model=full_model_name,
                    contents=contents,
                    config=config
                )
                return response.text

        except Exception as e:
            error_msg = f"[gemini_basic_call] Gemini basic call failed: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise Exception(error_msg) from e

    def gemini_tool_call(self, prompt: str = None, messages: List[Dict] = None, tools: List[Dict] = None,
                        model: str = "gemini-2.5-flash", system_prompt: Optional[str] = None,
                        stream: bool = False, on_token: Optional[Callable[[str], None]] = None,
                        on_reasoning: Optional[Callable[[str], None]] = None,
                        reasoning_effort: str = "medium",
                        **kwargs) -> Dict:
        """Gemini API call with tool/function calling support, streaming, and thinking mode.

        Args:
            prompt: Optional single user prompt (legacy)
            messages: Full conversation history as list of message dicts
            tools: List of tool definitions
            model: Model name (gemini-2.5-flash or gemini-2.5-pro)
            system_prompt: Optional system prompt
            stream: Enable streaming response
            on_token: Callback for each content token during streaming
            on_reasoning: Callback for reasoning/thinking tokens during streaming
            reasoning_effort: Reasoning effort level (none/low/medium/high) - maps to thinking_budget
            **kwargs: Additional parameters (unused)

        Returns:
            Dict with 'content' and 'tool_calls' keys
        
        Tools schema (Gemini FunctionDeclaration):
            tools = [
                {
                    "name": "get_weather",
                    "description": "Retrieve current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name, e.g., Boston"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["location"]
                    }
                }
            ]

        Also accepted (will be normalized to FunctionDeclaration):
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Retrieve current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                            },
                            "required": ["location"]
                        }
                    }
                }
            ]

        Response shape (normalized by this method):
            {
                "content": "...",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\":\"Boston\",\"unit\":\"celsius\"}"
                        }
                    }
                ]
            }
        """
        if not self.gemini_client:
            raise ValueError("Gemini client not initialized. Check GOOGLE_API_KEY.")

        # Get the full model name from mapping
        full_model_name = self.gemini_models.get(model, model)

        # Build the content for Gemini
        if messages:
            # Convert messages to a single prompt string for Gemini
            combined_prompt = ""
            if system_prompt:
                combined_prompt += f"System: {system_prompt}\n\n"

            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    if role == "system":
                        combined_prompt += f"System: {content}\n\n"
                    elif role == "user":
                        combined_prompt += f"User: {content}\n\n"
                    elif role == "assistant":
                        combined_prompt += f"Assistant: {content}\n\n"
                    elif role == "tool":
                        # Include tool responses in context
                        tool_name = msg.get('name', 'unknown')
                        combined_prompt += f"Tool ({tool_name}): {content}\n\n"

            prompt_content = combined_prompt.strip()
        elif prompt:
            # Legacy single prompt
            if system_prompt:
                prompt_content = f"{system_prompt}\n\n{prompt}"
            else:
                prompt_content = prompt
        else:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

        try:
            # Validate and normalize tools
            if tools is None:
                raise ValueError("Tools must be provided")
            if not isinstance(tools, list):
                raise ValueError("Tools must be provided as a list of dictionaries")

            normalized_funcs: List[Dict[str, Any]] = []
            for tool in tools:
                # Case 1: Tool already in the minimal FunctionDeclaration shape
                if isinstance(tool, dict) and "name" in tool:
                    normalized_funcs.append(tool)
                # Case 2: Tool follows Fireworks/JSON schema {"type":"function", "function": {...}}
                elif isinstance(tool, dict) and tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                    fn_def = tool["function"]
                    if "name" not in fn_def:
                        raise ValueError("Function entry missing 'name' field")
                    normalized_funcs.append(fn_def)
                else:
                    raise ValueError("Invalid tool format supplied to gemini_tool_use")

            # Enhance prompt to ensure single function call
            enhanced_prompt = f"""
                {prompt_content}

                IMPORTANT: You must call exactly ONE function per response. Choose the most appropriate single function based on the current context and execute only that function. Do not call multiple functions in the same response.
            """

            # Create Tool object with the normalized function declarations
            tool_object = types.Tool(function_declarations=normalized_funcs)

            # Map reasoning_effort to thinking_budget
            thinking_budget_map = {
                "none": 0,        # No thinking/reasoning
                "low": 4096,      # Minimal thinking
                "medium": -1,     # Dynamic thinking (recommended)
                "high": 16384     # Extended thinking
            }
            thinking_budget = thinking_budget_map.get(reasoning_effort, -1)

            # Create config with tools and thinking support
            config = types.GenerateContentConfig(
                tools=[tool_object],
                thinking_config=types.ThinkingConfig(
                    thinking_budget=thinking_budget
                )
            )

            if stream:
                # Streaming response with tools and thinking
                response_stream = self.gemini_client.models.generate_content_stream(
                    model=full_model_name,
                    contents=enhanced_prompt,
                    config=config
                )

                full_text = []
                full_thoughts = []
                function_calls = []

                for chunk in response_stream:
                    if chunk.candidates and len(chunk.candidates) > 0:
                        candidate = chunk.candidates[0]
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                # Check if this is a thought (reasoning) part
                                if hasattr(part, 'thought') and part.thought:
                                    # This is reasoning/thinking content
                                    if hasattr(part, 'text') and part.text:
                                        full_thoughts.append(part.text)
                                        if on_reasoning:
                                            try:
                                                on_reasoning(part.text)
                                            except Exception:
                                                pass
                                # Handle text
                                elif hasattr(part, 'text') and part.text:
                                    full_text.append(part.text)
                                    if on_token:
                                        try:
                                            on_token(part.text)
                                        except Exception:
                                            pass
                                # Handle function calls
                                elif hasattr(part, 'function_call') and part.function_call:
                                    fc = {
                                        "type": "function",
                                        "function": {
                                            "name": part.function_call.name,
                                            "arguments": json.dumps(dict(part.function_call.args)) if part.function_call.args else "{}"
                                        }
                                    }
                                    if fc not in function_calls:
                                        function_calls.append(fc)

                return {
                    "content": ''.join(full_text),
                    "tool_calls": function_calls
                }
            else:
                # Non-streaming response
                response = self.gemini_client.models.generate_content(
                    model=full_model_name,
                    contents=enhanced_prompt,
                    config=config
                )

                # Extract function calls and text
                function_calls = []
                response_text = ""

                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            # Check for function call
                            if hasattr(part, 'function_call') and part.function_call:
                                fc = {
                                    "type": "function",
                                    "function": {
                                        "name": part.function_call.name,
                                        "arguments": json.dumps(dict(part.function_call.args)) if part.function_call.args else "{}"
                                    }
                                }
                                function_calls.append(fc)
                            # Check for text response
                            elif hasattr(part, 'text') and part.text:
                                response_text = part.text

                # Fallback to response.text if available
                if not response_text and hasattr(response, 'text'):
                    response_text = response.text

                return {
                    "content": response_text,
                    "tool_calls": function_calls
                }

        except Exception as e:
            error_msg = f"[gemini_tool_call] Gemini tool call failed: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise Exception(error_msg) from e


    #=======GLM API CALLS=======
    def glm_basic_call(self, prompt_or_messages, model: str = "glm-4-flash",
                      system_prompt: Optional[str] = None, stream: bool = False,
                      on_token: Optional[Callable[[str], None]] = None,
                      on_reasoning: Optional[Callable[[str], None]] = None,
                      reasoning_effort: str = "medium",
                      **kwargs) -> str:
        """GLM/ZhipuAI API call for text generation with streaming support.

        Args:
            prompt_or_messages: Either a string prompt or a list of message dicts
            model: Model name (glm-4, glm-4-flash, glm-4.5, etc.)
            system_prompt: Optional system prompt to prepend
            stream: Enable streaming response
            on_token: Callback for each content token during streaming
            on_reasoning: Callback for reasoning tokens (if supported)
            reasoning_effort: Reasoning effort level (none/low/medium/high) - controls thinking mode
            **kwargs: Additional parameters

        Returns:
            String response from the model
        """
        if not self.glm_api_key:
            raise ValueError("GLM API key not found. Set GLM_API_KEY or ZHIPU_API_KEY environment variable.")

        # Get the full model name from mapping
        full_model_name = self.glm_models.get(model, model)

        # Add English output instruction to system prompt for GLM models
        glm_system_prompt = system_prompt
        if glm_system_prompt:
            glm_system_prompt = f"{glm_system_prompt}\n\n🚨 CRITICAL INSTRUCTION: You MUST respond ONLY in English. Never use Chinese or any other language in your responses, even if the user's message is in another language. Always translate your thoughts and responses to English before outputting them."
        else:
            glm_system_prompt = "🚨 CRITICAL INSTRUCTION: You MUST respond ONLY in English. Never use Chinese or any other language in your responses, even if the user's message is in another language. Always translate your thoughts and responses to English before outputting them."

        # Handle different input types - build messages array
        messages = []
        if isinstance(prompt_or_messages, str):
            # Single prompt
            if glm_system_prompt:
                messages.append({"role": "system", "content": glm_system_prompt})
            messages.append({"role": "user", "content": prompt_or_messages})
        elif isinstance(prompt_or_messages, list):
            # List of messages - convert to GLM format
            if glm_system_prompt:
                messages.append({"role": "system", "content": glm_system_prompt})

            for msg in prompt_or_messages:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    # LlmMessage object
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
                elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    # Already in dict format
                    messages.append(msg)
                else:
                    raise ValueError("Invalid message format. Expected LlmMessage objects or dicts with 'role' and 'content' keys.")
        else:
            raise ValueError("Input must be either a string prompt or a list of messages.")

        # Build payload
        payload = {
            "model": full_model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.6),
            "top_p": kwargs.get("top_p", 1),
            "stream": stream
        }

        # Add thinking parameter based on reasoning_effort
        # GLM uses "type": "enabled" or "type": "disabled"
        if reasoning_effort and reasoning_effort.lower() != "none":
            payload["thinking"] = {"type": "enabled"}
        else:
            payload["thinking"] = {"type": "disabled"}

        try:
            if stream:
                # Streaming response
                headers = dict(self.glm_headers)
                headers["Accept"] = "text/event-stream"

                full_text = []
                full_reasoning = []
                with requests.post(
                    self.glm_endpoint,
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=kwargs.get("timeout_seconds", 120)
                ) as response:
                    response.raise_for_status()

                    for line in response.iter_lines(decode_unicode=True):
                        if line and line.startswith("data: "):
                            if line == "data: [DONE]":
                                break
                            try:
                                json_data = json.loads(line[6:])  # Remove "data: " prefix
                                if "choices" in json_data and len(json_data["choices"]) > 0:
                                    delta = json_data["choices"][0].get("delta", {})

                                    # Handle reasoning_content streaming (comes first)
                                    if "reasoning_content" in delta and delta["reasoning_content"]:
                                        reason_chunk = delta["reasoning_content"]
                                        full_reasoning.append(reason_chunk)
                                        if on_reasoning:
                                            try:
                                                on_reasoning(reason_chunk)
                                            except Exception:
                                                pass

                                    # Handle content streaming (comes after reasoning)
                                    if "content" in delta and delta["content"]:
                                        text_chunk = delta["content"]
                                        full_text.append(text_chunk)
                                        if on_token:
                                            try:
                                                on_token(text_chunk)
                                            except Exception:
                                                pass
                            except json.JSONDecodeError:
                                continue

                return ''.join(full_text)
            else:
                # Non-streaming response
                response = requests.post(
                    self.glm_endpoint,
                    headers=self.glm_headers,
                    json=payload,
                    timeout=kwargs.get("timeout_seconds", 120)
                )
                response.raise_for_status()

                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    message_obj = result["choices"][0].get("message", {})
                    return message_obj.get("content", "")
                else:
                    error_msg = "No response content received from GLM API"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f"[glm_basic_call] GLM API request failed: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise Exception(error_msg) from e
        except Exception as e:
            error_msg = f"[glm_basic_call] GLM basic call failed: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise Exception(error_msg) from e

    def glm_tool_call(self, prompt: str = None, messages: List[Dict] = None, tools: List[Dict] = None,
                     model: str = "glm-4-flash", system_prompt: Optional[str] = None,
                     stream: bool = False, on_token: Optional[Callable[[str], None]] = None,
                     on_reasoning: Optional[Callable[[str], None]] = None,
                     reasoning_effort: str = "medium",
                     **kwargs) -> Dict:
        """GLM/ZhipuAI API call with tool/function calling support and streaming.

        Args:
            prompt: Optional single user prompt (legacy)
            messages: Full conversation history as list of message dicts
            tools: List of tool definitions
            model: Model name (glm-4, glm-4-flash, etc.)
            system_prompt: Optional system prompt
            stream: Enable streaming response
            on_token: Callback for each content token during streaming
            on_reasoning: Callback for reasoning tokens (if supported)
            reasoning_effort: Reasoning effort level (none/low/medium/high) - controls thinking mode
            **kwargs: Additional parameters

        Returns:
            Dict with 'content' and 'tool_calls' keys
        
        Tools schema (GLM chat completions):
            payload extras when tools are provided:
                {
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Retrieve current weather for a city.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "location": {"type": "string", "description": "City name"},
                                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                                    },
                                    "required": ["location"]
                                }
                            }
                        }
                    ],
                    "tool_choice": "auto"
                }

        Response shape (non-streaming):
            {
                "content": "...",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\":\"Boston\",\"unit\":\"celsius\"}"
                        }
                    }
                ]
            }

        Streaming responses deliver incremental tool calls via choices[0].delta.tool_calls with appended arguments; this method accumulates and returns the consolidated list.
        """
        self.logger.info(f"🚀 Starting GLM tool call with model: {model}")

        if not self.glm_api_key:
            self.logger.error("❌ GLM API key not found")
            raise ValueError("GLM API key not found. Set GLM_API_KEY or ZHIPU_API_KEY environment variable.")

        # Get the full model name from mapping
        full_model_name = self.glm_models.get(model, model)
        self.logger.debug(f"📋 Using model: {full_model_name}")

        # Add English output instruction to system prompt for GLM models
        glm_system_prompt = system_prompt
        if glm_system_prompt:
            glm_system_prompt = f"{glm_system_prompt}\n\n🚨 CRITICAL INSTRUCTION: You MUST respond ONLY in English. Never use Chinese or any other language in your responses, even if the user's message is in another language. Always translate your thoughts and responses to English before outputting them."
        else:
            glm_system_prompt = "🚨 CRITICAL INSTRUCTION: You MUST respond ONLY in English. Never use Chinese or any other language in your responses, even if the user's message is in another language. Always translate your thoughts and responses to English before outputting them."

        # Build messages array
        api_messages = []
        if glm_system_prompt:
            api_messages.append({"role": "system", "content": glm_system_prompt})

        if messages:
            # Use provided conversation history
            for msg in messages:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    # LlmMessage object
                    api_messages.append({"role": msg.role, "content": msg.content})
                elif isinstance(msg, dict):
                    api_messages.append(msg)
                else:
                    raise ValueError("Invalid message format in glm_tool_call")
        elif prompt:
            # Legacy single prompt
            api_messages.append({"role": "user", "content": prompt})
        else:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

        # Build payload
        payload = {
            "model": full_model_name,
            "messages": api_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.3),
            "stream": stream
        }

        # Add thinking parameter based on reasoning_effort
        # GLM uses "type": "enabled" or "type": "disabled"
        if reasoning_effort and reasoning_effort.lower() != "none":
            payload["thinking"] = {"type": "enabled"}
        else:
            payload["thinking"] = {"type": "disabled"}

        # Add tools if provided
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            if stream:
                # Streaming response with tools
                headers = dict(self.glm_headers)
                headers["Accept"] = "text/event-stream"

                content_chunks = []
                reasoning_chunks = []
                tool_calls_accumulated = []

                with requests.post(
                    self.glm_endpoint,
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=kwargs.get("timeout_seconds", 120)
                ) as response:
                    response.raise_for_status()

                    for line in response.iter_lines(decode_unicode=True):
                        if line and line.startswith("data: "):
                            if line == "data: [DONE]":
                                break
                            try:
                                json_data = json.loads(line[6:])
                                if "choices" in json_data and len(json_data["choices"]) > 0:
                                    delta = json_data["choices"][0].get("delta", {})

                                    # Handle reasoning_content streaming (comes first)
                                    if "reasoning_content" in delta and delta["reasoning_content"]:
                                        reason_chunk = delta["reasoning_content"]
                                        reasoning_chunks.append(reason_chunk)
                                        if on_reasoning:
                                            try:
                                                on_reasoning(reason_chunk)
                                            except Exception:
                                                pass

                                    # Handle content streaming (comes after reasoning)
                                    if "content" in delta and delta["content"]:
                                        text_chunk = delta["content"]
                                        content_chunks.append(text_chunk)
                                        if on_token:
                                            try:
                                                on_token(text_chunk)
                                            except Exception:
                                                pass

                                    # Handle tool calls
                                    if "tool_calls" in delta:
                                        for tc in delta["tool_calls"]:
                                            if "index" in tc:
                                                idx = tc["index"]
                                                # Ensure we have enough slots
                                                while len(tool_calls_accumulated) <= idx:
                                                    tool_calls_accumulated.append({})
                                                # Merge the delta
                                                if "id" in tc:
                                                    tool_calls_accumulated[idx]["id"] = tc["id"]
                                                if "type" in tc:
                                                    tool_calls_accumulated[idx]["type"] = tc["type"]
                                                if "function" in tc:
                                                    if "function" not in tool_calls_accumulated[idx]:
                                                        tool_calls_accumulated[idx]["function"] = {}
                                                    if "name" in tc["function"]:
                                                        tool_calls_accumulated[idx]["function"]["name"] = tc["function"]["name"]
                                                    if "arguments" in tc["function"]:
                                                        if "arguments" not in tool_calls_accumulated[idx]["function"]:
                                                            tool_calls_accumulated[idx]["function"]["arguments"] = ""
                                                        tool_calls_accumulated[idx]["function"]["arguments"] += tc["function"]["arguments"]
                            except json.JSONDecodeError:
                                continue

                return {
                    "content": ''.join(content_chunks),
                    "tool_calls": tool_calls_accumulated
                }
            else:
                # Non-streaming response
                timeout = kwargs.get("timeout_seconds", 120)  # Increased back to 120
                self.logger.debug(f"🌐 Sending request to GLM API (timeout: {timeout}s)")

                try:
                    response = requests.post(
                        self.glm_endpoint,
                        headers=self.glm_headers,
                        json=payload,
                        timeout=timeout
                    )
                    self.logger.debug(f"✅ Response received: {response.status_code}")
                    response.raise_for_status()
                except requests.exceptions.Timeout:
                    self.logger.error(f"⏰ Request timed out after {timeout}s")
                    raise
                except requests.exceptions.ConnectionError as e:
                    self.logger.error(f"🔌 Connection error: {str(e)[:100]}")
                    raise

                response_json = response.json()
                message = response_json["choices"][0]["message"]

                return {
                    "content": message.get("content", ""),
                    "tool_calls": message.get("tool_calls", [])
                }

        except requests.exceptions.RequestException as e:
            error_msg = f"[glm_tool_call] GLM API request failed: {str(e)}"
            self.logger.error(f"❌ Request failed: {str(e)[:200]}")

            # If it's a timeout, try once more with a longer timeout
            if "timed out" in str(e).lower() and kwargs.get("timeout_seconds", 120) < 180:
                self.logger.warning("🔄 GLM timed out, retrying with longer timeout...")
                try:
                    return self.glm_tool_call(
                        prompt=prompt,
                        messages=messages,
                        tools=tools,
                        model=model,
                        system_prompt=system_prompt,
                        stream=stream,
                        on_token=on_token,
                        reasoning_effort=reasoning_effort,
                        timeout_seconds=180  # 3 minutes for retry
                    )
                except Exception:
                    self.logger.error("❌ Retry also failed")

            # Log the error without attempting fallback
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")

            raise Exception(error_msg) from e
        except Exception as e:
            error_msg = f"[glm_tool_call] GLM tool call failed: {str(e)}"
            self.logger.error(f"❌ Unexpected error: {str(e)[:200]}")
            raise Exception(error_msg) from e


    #=======FIREWORKS API CALLS=======
    def fw_basic_call(self, prompt_or_messages, model: Optional[str] = None, system_prompt: Optional[str] = None, stream: bool = False, on_token: Optional[Callable[[str], None]] = None, timeout_seconds: int = 120, reasoning_effort: Optional[str] = None, on_reasoning: Optional[Callable[[str], None]] = None, **kwargs) -> str:
        """Basic Fireworks AI API call for text generation. Accepts either a string prompt or a list of messages."""
        if not self.fireworks_api_key:
            raise ValueError("Fireworks API key not found. Check FIREWORKS_API_KEY environment variable.")

        # Use defaults if not provided
        if model is None:
            model = self.default_model
        if reasoning_effort is None:
            reasoning_effort = self.default_reasoning_effort

        # Get the full model name from mapping
        full_model_name = self.fireworks_models.get(model, model)

        # Handle different input types
        messages = []
        if isinstance(prompt_or_messages, str):
            # Single prompt - create user message
            messages = [{"role": "user", "content": prompt_or_messages}]
        elif isinstance(prompt_or_messages, list):
            # List of messages - convert to Fireworks format
            for msg in prompt_or_messages:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    # LlmMessage object - role is already a string
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
                elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    # Already in dict format
                    messages.append(msg)
                else:
                    raise ValueError("Invalid message format. Expected LlmMessage objects or dicts with 'role' and 'content' keys.")
        else:
            raise ValueError("Input must be either a string prompt or a list of messages.")

        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        # Build payload for chat completions API
        payload = {
            "model": full_model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 20480),
            "temperature": kwargs.get("temperature", 0.6),
            "top_p": kwargs.get("top_p", 1),
            "top_k": kwargs.get("top_k", 40),
            "presence_penalty": kwargs.get("presence_penalty", 0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0),
            "stream": stream
        }

        # Include reasoning_effort ONLY for reasoning-capable models (e.g., deepseek-r1, deepseek-v3p1 on Fireworks)
        supports_reasoning = model in {"deepseek-r1", "deepseek-v3p1"}
        if supports_reasoning and reasoning_effort and reasoning_effort != "none":
            payload["reasoning_effort"] = reasoning_effort

        if not stream:
            # Non-streaming request
            try:
                response = requests.post(
                    self.fireworks_endpoint,
                    headers=self.fireworks_headers,
                    data=json.dumps(payload),
                    timeout=timeout_seconds
                )
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        message_obj = result["choices"][0].get("message", {})
                        content = message_obj.get("content", "")
                        reasoning_content = message_obj.get("reasoning_content", "")

                        # Fallback: parse <think> ... </think> if present in content
                        if not reasoning_content and isinstance(content, str) and "<think>" in content and "</think>" in content:
                            try:
                                start_idx = content.find("<think>") + len("<think>")
                                end_idx = content.find("</think>")
                                extracted_reasoning = content[start_idx:end_idx]
                                # Remove the <think>...</think> block from visible content
                                visible_content = content[end_idx + len("</think>"):].lstrip("\n")
                                reasoning_content = extracted_reasoning
                                content = visible_content
                            except Exception:
                                pass

                        # Return raw content without formatting
                        return content
                    else:
                        error_msg = f"No response content received from Fireworks API. Response keys: {list(result.keys()) if result else 'Empty result'}"
                        self.logger.error(error_msg)
                        raise Exception(error_msg)
                else:
                    error_msg = f"Fireworks API request failed with status {response.status_code}: {response.text}"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)
            except requests.RequestException as e:
                error_msg = f"[fw_basic_call] Fireworks API request failed: {str(e)}"
                self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
                raise Exception(error_msg) from e
            except json.JSONDecodeError as e:
                error_msg = f"[fw_basic_call] Failed to parse Fireworks API response: {str(e)}"
                self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
                raise Exception(error_msg) from e
        
        # Streaming request using chat completions API
        headers = dict(self.fireworks_headers)
        headers["Accept"] = "text/event-stream"
        
        final_text_chunks: List[str] = []
        final_reasoning_chunks: List[str] = []
        try:
            with requests.post(
                self.fireworks_endpoint,
                headers=headers,
                data=json.dumps(payload),
                stream=True,
                timeout=timeout_seconds
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines(decode_unicode=True):
                    if line and line.startswith("data: "):
                        if line == "data: [DONE]":
                            break
                        try:
                            json_data = json.loads(line[6:])  # Remove "data: " prefix
                            if "choices" in json_data and len(json_data["choices"]) > 0:
                                delta = json_data["choices"][0].get("delta", {})

                                # Handle reasoning_content (comes first in the stream)
                                if "reasoning_content" in delta and delta["reasoning_content"]:
                                    reason_chunk = delta["reasoning_content"]
                                    final_reasoning_chunks.append(reason_chunk)
                                    if on_reasoning:
                                        try:
                                            on_reasoning(reason_chunk)
                                        except Exception:
                                            pass

                                # Handle content (comes after reasoning is complete)
                                if "content" in delta and delta["content"]:
                                    text_chunk = delta["content"]
                                    final_text_chunks.append(text_chunk)
                                    if on_token:
                                        try:
                                            on_token(text_chunk)
                                        except Exception:
                                            pass
                        except json.JSONDecodeError:
                            continue
        except requests.exceptions.RequestException as e:
            error_msg = f"[fw_basic_call(stream)] Fireworks streaming failed: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise Exception(error_msg) from e
        
        combined_visible = "".join(final_text_chunks)
        if not final_reasoning_chunks and ("<think>" in combined_visible and "</think>" in combined_visible):
            try:
                start_idx = combined_visible.find("<think>") + len("<think>")
                end_idx = combined_visible.find("</think>")
                extracted_reasoning = combined_visible[start_idx:end_idx]
                combined_visible = combined_visible[end_idx + len("</think>"):].lstrip("\n")
                final_reasoning_chunks.append(extracted_reasoning)
            except Exception:
                pass

        # Return raw content without formatting
        return combined_visible




    def fw_tool_call(self, prompt: str = None, messages: List[LlmMessage] = None, tools: List[FireworksTool] = None,
                                model_key: str = "deepseek-v3p1", max_tokens: int = 4096,
                                temperature: float = 0.3, system_prompt: Optional[str] = None,
                                stream: bool = False, on_token: Optional[Callable[[str], None]] = None,
                                on_reasoning: Optional[Callable[[str], None]] = None,
                                reasoning_effort: Optional[str] = None,
                                timeout_seconds: int = 120) -> FireworksToolCallResponse:
        """
        Call Fireworks AI with tool/function calling support.

        Args:
            prompt: Optional single user prompt (legacy, use messages instead)
            messages: Optional full conversation history as list of LlmMessage objects
            tools: List of tool definitions in Fireworks format, e.g.:
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_financial_data",
                            "description": "Get financial data for a company",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "company": {"type": "string", "description": "Company name"},
                                    "metric": {"type": "string", "enum": ["revenue", "net_income"]},
                                    "year": {"type": "integer", "description": "Financial year"}
                                },
                                "required": ["company", "metric", "year"]
                            }
                        }
                    }
                ]
            model_key: Model key from available Fireworks models
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            stream: Enable streaming response
            on_token: Callback for each content token during streaming
            on_reasoning: Callback for each reasoning token during streaming
            reasoning_effort: Reasoning effort level (low/medium/high) for supported models

        Returns:
            Dict with 'content' and 'tool_calls' keys.
            When streaming, 'content' contains the full accumulated response.
        """

        if not self.fireworks_api_key:
            raise ValueError("Fireworks API key not found. Check FIREWORKS_API_KEY environment variable.")

        if model_key not in self.fireworks_models:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(self.fireworks_models.keys())}")

        model = self.fireworks_models[model_key]

        # Validate tools format
        if not isinstance(tools, list):
            raise ValueError("Tools must be a list of tool definitions")

        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                raise ValueError("Each tool must be a dict with 'type': 'function'")
            if "function" not in tool or "name" not in tool["function"]:
                raise ValueError("Each tool must have a 'function' with a 'name'")

        # Build messages array - use provided messages or fallback to single prompt
        api_messages: List[Dict[str, Any]] = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        
        if messages:
            # Use provided conversation history - convert LlmMessage objects to dicts
            for msg in messages:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    api_messages.append({"role": msg.role, "content": msg.content})
                elif isinstance(msg, dict):
                    api_messages.append(msg)
                else:
                    raise ValueError("Invalid message format in fw_tool_call")
        elif prompt:
            # Fallback to single prompt for backward compatibility
            api_messages.append({"role": "user", "content": prompt})
        else:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
            "tools": tools,
            "tool_choice": "auto",
            "stream": stream
        }

        # Add reasoning_effort for supported models (similar to fw_basic_call)
        supports_reasoning = model_key in {"deepseek-r1", "deepseek-v3p1"}
        if supports_reasoning and reasoning_effort and reasoning_effort != "none":
            payload["reasoning_effort"] = reasoning_effort

        if not stream:
            # Non-streaming request
            try:
                response = requests.post(
                    self.fireworks_endpoint,
                    headers=self.fireworks_headers,
                    json=payload,
                    timeout=timeout_seconds
                )
                response.raise_for_status()
                response_json = response.json()
                message = response_json["choices"][0]["message"]

                content = message.get("content", "")
                
                return {
                    "content": content,
                    "tool_calls": message.get("tool_calls", [])
                }
            except requests.exceptions.RequestException as e:
                error_msg = f"[fw_tool_call] Fireworks function calling failed: {str(e)}"
                self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
                raise Exception(error_msg) from e
            except json.JSONDecodeError as e:
                error_msg = f"[fw_tool_call] Failed to parse Fireworks API response: {str(e)}"
                self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
                raise Exception(error_msg) from e
            except (KeyError, IndexError) as e:
                error_msg = f"[fw_tool_call] Failed to extract data from Fireworks response: {str(e)}"
                self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
                raise Exception(error_msg) from e
        
        # Streaming request
        headers = dict(self.fireworks_headers)
        headers["Accept"] = "text/event-stream"
        
        content_chunks: List[str] = []
        reasoning_chunks: List[str] = []
        tool_calls_accumulated = []

        try:
            with requests.post(
                self.fireworks_endpoint,
                headers=headers,
                json=payload,
                stream=True,
                timeout=timeout_seconds
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines(decode_unicode=True):
                    if line and line.startswith("data: "):
                        if line == "data: [DONE]":
                            break
                        try:
                            json_data = json.loads(line[6:])  # Remove "data: " prefix
                            if "choices" in json_data and len(json_data["choices"]) > 0:
                                delta = json_data["choices"][0].get("delta", {})

                                # Handle reasoning_content (comes first in the stream)
                                if "reasoning_content" in delta and delta["reasoning_content"]:
                                    reason_chunk = delta["reasoning_content"]
                                    reasoning_chunks.append(reason_chunk)
                                    if on_reasoning:
                                        try:
                                            on_reasoning(reason_chunk)
                                        except Exception:
                                            pass

                                # Handle content streaming (comes after reasoning)
                                if "content" in delta and delta["content"]:
                                    text_chunk = delta["content"]
                                    content_chunks.append(text_chunk)
                                    if on_token:
                                        try:
                                            on_token(text_chunk)
                                        except Exception:
                                            pass
                                
                                # Handle tool calls - they may come incrementally
                                if "tool_calls" in delta:
                                    # Tool calls might be partial, need to accumulate properly
                                    for tc in delta["tool_calls"]:
                                        if "index" in tc:
                                            idx = tc["index"]
                                            # Ensure we have enough slots
                                            while len(tool_calls_accumulated) <= idx:
                                                tool_calls_accumulated.append({})
                                            # Merge the delta into the accumulated tool call
                                            if "id" in tc:
                                                tool_calls_accumulated[idx]["id"] = tc["id"]
                                            if "type" in tc:
                                                tool_calls_accumulated[idx]["type"] = tc["type"]
                                            if "function" in tc:
                                                if "function" not in tool_calls_accumulated[idx]:
                                                    tool_calls_accumulated[idx]["function"] = {}
                                                if "name" in tc["function"]:
                                                    tool_calls_accumulated[idx]["function"]["name"] = tc["function"]["name"]
                                                if "arguments" in tc["function"]:
                                                    if "arguments" not in tool_calls_accumulated[idx]["function"]:
                                                        tool_calls_accumulated[idx]["function"]["arguments"] = ""
                                                    tool_calls_accumulated[idx]["function"]["arguments"] += tc["function"]["arguments"]
                        except json.JSONDecodeError:
                            continue
            
            # Debug log for tool calls
            if tool_calls_accumulated:
                self.logger.debug(f"[fw_tool_call streaming] Accumulated tool calls: {json.dumps(tool_calls_accumulated)[:500]}")
            
            return {
                "content": "".join(content_chunks),
                "tool_calls": tool_calls_accumulated
            }

        except requests.exceptions.RequestException as e:
            error_msg = f"[fw_tool_call(stream)] Fireworks streaming failed: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise Exception(error_msg) from e


    def format_function_calls(self, model: str, functions: List[Dict]) -> List[Dict]:
        """
        Convert tool definitions to the appropriate format for the specified model provider.

        Each LLM provider has different tool calling formats:
        - Gemini: Uses FunctionDeclaration format with just name, description, parameters
        - Fireworks: Uses OpenAI-compatible format with type: "function" wrapper
        - GLM: Uses OpenAI-compatible format with type: "function" wrapper

        Args:
            model: Model name (auto-detects provider)
            functions: List of tool definitions in standard format

        Standard format expected:
            [
                {
                    "name": "function_name",
                    "description": "Function description",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "param1": {"type": "string", "description": "Param description"},
                            "param2": {"type": "integer", "description": "Another param"}
                        },
                        "required": ["param1"]
                    }
                }
            ]

        Returns:
            List of tool definitions in the model provider's expected format
        """
        if not functions or not isinstance(functions, list):
            return []

        # Determine provider from model
        if model in self.gemini_models or any(model in v for v in self.gemini_models.values()) or any(keyword in model.lower() for keyword in ['gemini', 'google']):
            # Gemini FunctionDeclaration format (no wrapper)
            formatted_tools = []
            for func in functions:
                if isinstance(func, dict):
                    # Already in FunctionDeclaration format
                    if "name" in func and "description" in func:
                        formatted_tools.append(func)
                    # Convert from OpenAI format
                    elif func.get("type") == "function" and "function" in func:
                        formatted_tools.append(func["function"])
                    else:
                        # Try to construct from basic fields
                        if "name" in func:
                            formatted_tool = {
                                "name": func["name"],
                                "description": func.get("description", ""),
                                "parameters": func.get("parameters", {"type": "object", "properties": {}, "required": []})
                            }
                            formatted_tools.append(formatted_tool)
            return formatted_tools

        elif model in self.fireworks_models or any(model in v for v in self.fireworks_models.values()):
            # Fireworks/OpenAI format with type wrapper
            formatted_tools = []
            for func in functions:
                if isinstance(func, dict):
                    # Already in OpenAI format
                    if func.get("type") == "function" and "function" in func:
                        formatted_tools.append(func)
                    else:
                        # Convert to OpenAI format
                        if "name" in func:
                            formatted_tool = {
                                "type": "function",
                                "function": {
                                    "name": func["name"],
                                    "description": func.get("description", ""),
                                    "parameters": func.get("parameters", {"type": "object", "properties": {}, "required": []})
                                }
                            }
                            formatted_tools.append(formatted_tool)
            return formatted_tools

        elif model in self.glm_models or any(model in v for v in self.glm_models.values()) or any(keyword in model.lower() for keyword in ['glm', 'zhipu', 'chatglm']):
            # GLM uses OpenAI-compatible format with type wrapper (same as Fireworks)
            formatted_tools = []
            for func in functions:
                if isinstance(func, dict):
                    # Already in OpenAI format
                    if func.get("type") == "function" and "function" in func:
                        formatted_tools.append(func)
                    else:
                        # Convert to OpenAI format
                        if "name" in func:
                            formatted_tool = {
                                "type": "function",
                                "function": {
                                    "name": func["name"],
                                    "description": func.get("description", ""),
                                    "parameters": func.get("parameters", {"type": "object", "properties": {}, "required": []})
                                }
                            }
                            formatted_tools.append(formatted_tool)
            return formatted_tools

        else:
            # Default to OpenAI/Fireworks format for unknown models
            self.logger.warning(f"[format_function_calls] Unknown model '{model}', defaulting to OpenAI format")
            formatted_tools = []
            for func in functions:
                if isinstance(func, dict):
                    # Already in OpenAI format
                    if func.get("type") == "function" and "function" in func:
                        formatted_tools.append(func)
                    else:
                        # Convert to OpenAI format
                        if "name" in func:
                            formatted_tool = {
                                "type": "function",
                                "function": {
                                    "name": func["name"],
                                    "description": func.get("description", ""),
                                    "parameters": func.get("parameters", {"type": "object", "properties": {}, "required": []})
                                }
                            }
                            formatted_tools.append(formatted_tool)
            return formatted_tools


    #=======UNIFIED LLM CALL FUNCTION=======
    def call(self, prompt_or_messages=None, model: Optional[str] = None,
             reasoning_effort: str = "medium", tools: Optional[List[Dict]] = None,
             system_prompt: Optional[str] = None, stream: bool = False,
             on_token: Optional[Callable[[str], None]] = None,
             on_reasoning: Optional[Callable[[str], None]] = None,
             **kwargs) -> Union[str, Dict]:
        """
        Unified LLM call function that automatically routes to the appropriate provider.

        Args:
            prompt_or_messages: Either a string prompt or a list of message dicts
            model: Model name (auto-detects provider from model)
            reasoning_effort: Reasoning effort level (none/low/medium/high)
            tools: Optional list of tool definitions for function calling
            system_prompt: Optional system prompt
            stream: Enable streaming response
            on_token: Callback for each content token during streaming
            on_reasoning: Callback for reasoning/thinking tokens during streaming
            **kwargs: Additional provider-specific parameters

        Returns:
            String for basic calls, or Dict with 'content' and 'tool_calls' for tool calls
        """
        try:
            # Use defaults if not provided
            if model is None:
                model = self.default_model
            if reasoning_effort is None:
                reasoning_effort = self.default_reasoning_effort

            self.logger.info(f"LLM call initiated - model: {model}, reasoning_effort: {reasoning_effort}, tools: {'yes' if tools else 'no'}, stream: {stream}")

            # Format tools according to the model provider if tools are provided
            if tools:
                tools = self.format_function_calls(model, tools)

            # Determine provider and route to appropriate function
            if model in self.fireworks_models or any(model in v for v in self.fireworks_models.values()):
                # Fireworks AI
                self.logger.debug(f"Routing to Fireworks AI provider - tools: {'yes' if tools else 'no'}")
                if tools:
                    return self.fw_tool_call(
                        prompt=prompt_or_messages if isinstance(prompt_or_messages, str) else None,
                        messages=[prompt_or_messages] if isinstance(prompt_or_messages, str) else prompt_or_messages,
                        tools=tools,
                        model_key=model if model in self.fireworks_models else None,
                        system_prompt=system_prompt,
                        stream=stream,
                        on_token=on_token,
                        on_reasoning=on_reasoning,
                        reasoning_effort=reasoning_effort,
                        **kwargs
                    )
                else:
                    return self.fw_basic_call(
                        prompt_or_messages=prompt_or_messages,
                        model=model,
                        system_prompt=system_prompt,
                        stream=stream,
                        on_token=on_token,
                        on_reasoning=on_reasoning,
                        reasoning_effort=reasoning_effort,
                        **kwargs
                    )

            elif model in self.gemini_models or any(model in v for v in self.gemini_models.values()):
                # Google Gemini
                self.logger.debug(f"Routing to Google Gemini provider - tools: {'yes' if tools else 'no'}")
                if tools:
                    return self.gemini_tool_call(
                        prompt=prompt_or_messages if isinstance(prompt_or_messages, str) else None,
                        messages=[prompt_or_messages] if isinstance(prompt_or_messages, str) else prompt_or_messages,
                        tools=tools,
                        model=model,
                        system_prompt=system_prompt,
                        stream=stream,
                        on_token=on_token,
                        on_reasoning=on_reasoning,
                        reasoning_effort=reasoning_effort,
                        **kwargs
                    )
                else:
                    return self.gemini_basic_call(
                        prompt_or_messages=prompt_or_messages,
                        model=model,
                        system_prompt=system_prompt,
                        stream=stream,
                        on_token=on_token,
                        on_reasoning=on_reasoning,
                        reasoning_effort=reasoning_effort,
                        **kwargs
                    )

            elif model in self.glm_models or any(model in v for v in self.glm_models.values()):
                # GLM/ZhipuAI
                self.logger.debug(f"Routing to GLM/ZhipuAI provider - tools: {'yes' if tools else 'no'}")
                if tools:
                    return self.glm_tool_call(
                        prompt=prompt_or_messages if isinstance(prompt_or_messages, str) else None,
                        messages=prompt_or_messages if isinstance(prompt_or_messages, list) else None,
                        tools=tools,
                        model=model,
                        system_prompt=system_prompt,
                        stream=stream,
                        on_token=on_token,
                        on_reasoning=on_reasoning,
                        reasoning_effort=reasoning_effort,
                        **kwargs
                    )
                else:
                    return self.glm_basic_call(
                        prompt_or_messages=prompt_or_messages,
                        model=model,
                        system_prompt=system_prompt,
                        stream=stream,
                        on_token=on_token,
                        on_reasoning=on_reasoning,
                        reasoning_effort=reasoning_effort,
                        **kwargs
                    )

            else:
                # Try to infer from model name patterns if not found in mappings
                model_lower = model.lower()
                if any(keyword in model_lower for keyword in ['gemini', 'google']):
                    # Assume Gemini
                    self.logger.debug(f"Inferred Google Gemini provider from model name - tools: {'yes' if tools else 'no'}")
                    if tools:
                        return self.gemini_tool_call(
                            prompt=prompt_or_messages if isinstance(prompt_or_messages, str) else None,
                            messages=[prompt_or_messages] if isinstance(prompt_or_messages, str) else prompt_or_messages,
                            tools=tools,
                            model=model,
                            system_prompt=system_prompt,
                            stream=stream,
                            on_token=on_token,
                            on_reasoning=on_reasoning,
                            reasoning_effort=reasoning_effort,
                            **kwargs
                        )
                    else:
                        return self.gemini_basic_call(
                            prompt_or_messages=prompt_or_messages,
                            model=model,
                            system_prompt=system_prompt,
                            stream=stream,
                            on_token=on_token,
                            on_reasoning=on_reasoning,
                            reasoning_effort=reasoning_effort,
                            **kwargs
                        )

                elif any(keyword in model_lower for keyword in ['glm', 'zhipu', 'chatglm']):
                    # Assume GLM
                    self.logger.debug(f"Inferred GLM/ZhipuAI provider from model name - tools: {'yes' if tools else 'no'}")
                    if tools:
                        return self.glm_tool_call(
                            prompt=prompt_or_messages if isinstance(prompt_or_messages, str) else None,
                            messages=prompt_or_messages if isinstance(prompt_or_messages, list) else None,
                            tools=tools,
                            model=model,
                            system_prompt=system_prompt,
                            stream=stream,
                            on_token=on_token,
                            on_reasoning=on_reasoning,
                            reasoning_effort=reasoning_effort,
                            **kwargs
                        )
                    else:
                        return self.glm_basic_call(
                            prompt_or_messages=prompt_or_messages,
                            model=model,
                            system_prompt=system_prompt,
                            stream=stream,
                            on_token=on_token,
                            on_reasoning=on_reasoning,
                            reasoning_effort=reasoning_effort,
                            **kwargs
                        )

                else:
                    # Default to Fireworks for unknown models
                    self.logger.warning(f"Unknown model '{model}', defaulting to Fireworks AI provider - tools: {'yes' if tools else 'no'}")
                    if tools:
                        return self.fw_tool_call(
                            prompt=prompt_or_messages if isinstance(prompt_or_messages, str) else None,
                            messages=[prompt_or_messages] if isinstance(prompt_or_messages, str) else prompt_or_messages,
                            tools=tools,
                            model_key=model if model in self.fireworks_models else None,
                            system_prompt=system_prompt,
                            stream=stream,
                            on_token=on_token,
                            on_reasoning=on_reasoning,
                            reasoning_effort=reasoning_effort,
                            **kwargs
                        )
                    else:
                        return self.fw_basic_call(
                            prompt_or_messages=prompt_or_messages,
                            model=model,
                            system_prompt=system_prompt,
                            stream=stream,
                            on_token=on_token,
                            on_reasoning=on_reasoning,
                            reasoning_effort=reasoning_effort,
                            **kwargs
                        )

        except Exception as e:
            # Top-level error handling for the call() method
            error_msg = f"[call] LLM call failed for model '{model}': {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            # Re-raise with context
            raise Exception(error_msg) from e


# uv run python -m qrooper.agents.llm_calls
def _run_demo():
    """
    Demo function to test all QrooperLLM functionality.
    Tests basic calls, tool calls, and unified call method across all providers.
    """
    import os
    import logging

    # Configure logging to show all debug messages
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] [%(name)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    # Also configure the root logger to capture all messages
    logging.getLogger().setLevel(logging.DEBUG)

    print("🚀 Starting QrooperLLM Demo - Testing All Functionality\n")
    print("📝 Logging: DEBUG mode enabled - all LLM call details will be shown\n")

    # Initialize the LLM
    llm = QrooperLLM(desc="demo", model="deepseek-v3p1")

    # Check available API keys
    has_gemini = bool(os.getenv("GOOGLE_API_KEY"))
    has_fireworks = bool(os.getenv("FIREWORKS_API_KEY"))
    has_glm = bool(os.getenv("GLM_API_KEY") or os.getenv("ZHIPU_API_KEY"))

    print("📋 API Keys Status:")
    print(f"  Gemini (Google): {'✅' if has_gemini else '❌'}")
    print(f"  Fireworks AI: {'✅' if has_fireworks else '❌'}")
    print(f"  GLM/ZhipuAI: {'✅' if has_glm else '❌'}")
    print()

    # Define test tools once
    test_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform a mathematical calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                    },
                    "required": ["expression"]
                }
            }
        }
    ]

    print(f"\n📋 Available tools ({len(test_tools)}):")
    for tool in test_tools:
        name = tool.get('function', {}).get('name', 'Unknown')
        desc = tool.get('function', {}).get('description', 'No description')
        print(f"  • {name}: {desc}")

    # # ========================================
    # # GEMINI TESTS
    # # ========================================
    # print("\n" + "=" * 60)
    # print("🔷 GEMINI TESTS")
    # print("=" * 60)

    # if has_gemini:
    #     print("\n1.1 Gemini Basic Call...")
    #     print("  📤 Sending prompt: 'What is 2+2? Answer with just the number.'")
    #     print(f"  🔧 Model: gemini-2.5-flash | Reasoning: none")
    #     try:
    #         response = llm.gemini_basic_call(
    #             "What is 2+2? Answer with just the number.",
    #             model="gemini-2.5-flash",
    #             reasoning_effort="none"
    #         )
    #         print(f"\n  📥 RAW GEMINI RESPONSE:")
    #         print(f"  ─────────────────────────────")
    #         print(f"  {response}")
    #         print(f"  ─────────────────────────────")
    #         print(f"  ✅ Gemini Response: {response}")
    #     except Exception as e:
    #         print(f"  ❌ Gemini Basic Call Failed: {str(e)[:100]}...")

    #     print("\n1.2 Gemini Tool Call...")
    #     print("  📤 Sending prompt: 'What's the weather in New York?'")
    #     print(f"  🔧 Model: gemini-2.5-flash | Tools: 2 available")
    #     print("  📜 Tools being sent:")
    #     for tool in test_tools:
    #         print(f"    - {tool.get('function', {}).get('name', 'Unknown')}")
    #     try:
    #         response = llm.gemini_tool_call(
    #             prompt="What's the weather in New York?",
    #             tools=test_tools,
    #             model="gemini-2.5-flash",
    #             reasoning_effort="none"
    #         )
    #         print(f"\n  📥 RAW GEMINI TOOL RESPONSE:")
    #         print(f"  ─────────────────────────────")
    #         import json
    #         print(f"  {json.dumps(response, indent=2)}")
    #         print(f"  ─────────────────────────────")
    #         print(f"  ✅ Gemini Tool Response:")
    #         content = response.get('content', 'No content')
    #         print(f"    Content: {content[:200] if content else content}")
    #         print(f"    Tool Calls: {len(response.get('tool_calls', []))} calls")
    #         for i, tc in enumerate(response.get('tool_calls', [])):
    #             func_name = tc.get('function', {}).get('name', 'Unknown')
    #             func_args = tc.get('function', {}).get('arguments', '{}')
    #             print(f"      {i+1}. {func_name}({func_args})")
    #     except Exception as e:
    #         print(f"  ❌ Gemini Tool Call Failed: {str(e)[:200]}...")

    #     print("\n1.3 Gemini Thinking Mode...")
    #     print("  📤 Sending prompt: 'Solve this step by step: If 3x + 7 = 22, what is x?'")
    #     print(f"  🔧 Model: gemini-2.5-flash | Reasoning: medium | Stream: Enabled")
    #     try:
    #         reasoning_chunks = []
    #         reasoning_count = 0
    #         def on_reasoning(chunk):
    #             nonlocal reasoning_count
    #             reasoning_count += 1
    #             reasoning_chunks.append(chunk)
    #             print(f"  🤔 Reasoning chunk #{reasoning_count}:")
    #             print(f"      {chunk}")

    #         print("\n  🧠 Starting thinking mode...")
    #         response = llm.gemini_basic_call(
    #             "Solve this step by step: If 3x + 7 = 22, what is x?",
    #             model="gemini-2.5-flash",
    #             reasoning_effort="medium",
    #             stream=True,
    #             on_reasoning=on_reasoning,
    #             timeout_seconds=15
    #         )
    #         print(f"\n  📥 RAW GEMINI THINKING RESPONSE:")
    #         print(f"  ─────────────────────────────")
    #         print(f"  {response}")
    #         print(f"  ─────────────────────────────")
    #         print(f"  ✅ Thinking complete!")
    #         print(f"    Total reasoning chunks: {len(reasoning_chunks)}")
    #         print(f"    Combined reasoning: {''.join(reasoning_chunks)}")
    #         print(f"    Final answer: {response}")
    #     except Exception as e:
    #         print(f"  ❌ Gemini Thinking Failed: {str(e)[:200]}...")
    # else:
    #     print("\n⏭️ Skipping Gemini tests (no API key)")

    # ========================================
    # GLM TESTS
    # ========================================
    print("\n" + "=" * 60)
    print("🔶 GLM TESTS")
    print("=" * 60)

    if has_glm:
        print("\n2.1 GLM Basic Call...")
        print("  📤 Sending prompt: 'What is 4+4? Answer with just the number.'")
        print(f"  🔧 Model: glm-4.5-flash | Reasoning: none")
        try:
            response = llm.glm_basic_call(
                "What is 4+4? Answer with just the number.",
                model="glm-4.5-flash",
                reasoning_effort="none"
            )
            print(f"\n  📥 RAW GLM RESPONSE:")
            print(f"  ─────────────────────────────")
            print(f"  {response}")
            print(f"  ─────────────────────────────")
            print(f"  ✅ GLM Response: {response}")
        except Exception as e:
            print(f"  ❌ GLM Basic Call Failed: {str(e)[:100]}...")

        print("\n2.2 GLM Tool Call...")
        print("  📤 Sending prompt: 'What's the weather in London in Celsius?'")
        print(f"  🔧 Model: glm-4.5-flash | Tools: 2 available")
        print("  📜 Tools being sent:")
        for tool in test_tools:
            print(f"    - {tool.get('function', {}).get('name', 'Unknown')}")
        try:
            response = llm.glm_tool_call(
                prompt="What's the weather in London in Celsius?",
                tools=test_tools,
                model="glm-4.5-flash",
                reasoning_effort="none"
            )
            print(f"\n  📥 RAW GLM TOOL RESPONSE:")
            print(f"  ─────────────────────────────")
            import json
            print(f"  {json.dumps(response, indent=2)}")
            print(f"  ─────────────────────────────")
            print(f"  ✅ GLM Tool Response:")
            print(f"    Content: {response.get('content', 'No content')[:200]}...")
            print(f"    Tool Calls: {len(response.get('tool_calls', []))} calls")
            for i, tc in enumerate(response.get('tool_calls', [])):
                func_name = tc.get('function', {}).get('name', 'Unknown')
                func_args = tc.get('function', {}).get('arguments', '{}')
                print(f"      {i+1}. {func_name}({func_args})")
        except Exception as e:
            print(f"  ❌ GLM Tool Call Failed: {str(e)[:200]}...")
    else:
        print("\n⏭️ Skipping GLM tests (no API key)")

    # # ========================================
    # # FIREWORKS TESTS
    # # ========================================
    # print("\n" + "=" * 60)
    # print("🔥 FIREWORKS TESTS")
    # print("=" * 60)

    # if has_fireworks:
    #     print("\n3.1 Fireworks Basic Call...")
    #     print("  📤 Sending prompt: 'What is 3+3? Answer with just the number.'")
    #     print(f"  🔧 Model: deepseek-v3p1 | Reasoning: none")
    #     try:
    #         response = llm.fw_basic_call(
    #             "What is 3+3? Answer with just the number.",
    #             model="deepseek-v3p1",
    #             reasoning_effort="none"
    #         )
    #         print(f"\n  📥 RAW FIREWORKS RESPONSE:")
    #         print(f"  ─────────────────────────────")
    #         print(f"  {response}")
    #         print(f"  ─────────────────────────────")
    #         print(f"  ✅ Fireworks Response: {response}")
    #     except Exception as e:
    #         print(f"  ❌ Fireworks Basic Call Failed: {str(e)[:100]}...")

    #     print("\n3.2 Fireworks Tool Call...")
    #     print("  📤 Sending prompt: 'Calculate 25 * 4'")
    #     print(f"  🔧 Model: deepseek-v3p1 | Tools: 2 available")
    #     print("  📜 Tools being sent:")
    #     for tool in test_tools:
    #         print(f"    - {tool.get('function', {}).get('name', 'Unknown')}")
    #     try:
    #         response = llm.fw_tool_call(
    #             prompt="Calculate 25 * 4",
    #             tools=test_tools,
    #             model_key="deepseek-v3p1",
    #             reasoning_effort="none"
    #         )
    #         print(f"\n  📥 RAW FIREWORKS TOOL RESPONSE:")
    #         print(f"  ─────────────────────────────")
    #         import json
    #         print(f"  {json.dumps(response, indent=2)}")
    #         print(f"  ─────────────────────────────")
    #         print(f"  ✅ Fireworks Tool Response:")
    #         print(f"    Content: {response.get('content', 'No content')[:200]}...")
    #         print(f"    Tool Calls: {len(response.get('tool_calls', []))} calls")
    #         for i, tc in enumerate(response.get('tool_calls', [])):
    #             func_name = tc.get('function', {}).get('name', 'Unknown')
    #             func_args = tc.get('function', {}).get('arguments', '{}')
    #             print(f"      {i+1}. {func_name}({func_args})")
    #     except Exception as e:
    #         print(f"  ❌ Fireworks Tool Call Failed: {str(e)[:200]}...")

    #     print("\n3.3 Fireworks Streaming...")
    #     print("  📤 Sending prompt: 'Count from 1 to 5 slowly'")
    #     print(f"  🔧 Model: deepseek-v3p1 | Stream: Enabled")
    #     try:
    #         tokens = []
    #         token_count = 0
    #         def on_token(token):
    #             nonlocal token_count
    #             token_count += 1
    #             tokens.append(token)
    #             print(f"  📝 Token #{token_count}: '{token}'")
    #             print(f"      Accumulated: {''.join(tokens)[-50:]}...")

    #         print("\n  🌊 Starting stream...")
    #         response = llm.fw_basic_call(
    #             "Count from 1 to 5 slowly",
    #             model="deepseek-v3p1",
    #             stream=True,
    #             on_token=on_token,
    #             timeout_seconds=10
    #         )
    #         print(f"\n  📥 RAW FIREWORKS STREAMING RESPONSE:")
    #         print(f"  ─────────────────────────────")
    #         print(f"  {response}")
    #         print(f"  ─────────────────────────────")
    #         print(f"  ✅ Streaming complete!")
    #         print(f"    Total tokens received: {len(tokens)}")
    #         print(f"    Token list: {tokens}")
    #         print(f"    Full response: {response}")
    #     except Exception as e:
    #         print(f"  ❌ Fireworks Streaming Failed: {str(e)[:200]}...")
    # else:
    #     print("\n⏭️ Skipping Fireworks tests (no API key)")

    # # ========================================
    # # FORMATTING TESTS
    # # ========================================
    # print("\n" + "=" * 60)
    # print("🎨 TEST 4: TOOL FORMATTING")
    # print("=" * 60)

    # print("\n4.1 Testing tool format conversion for different providers...")

    # # Test with different model formats
    # standard_tools = [
    #     {
    #         "name": "test_function",
    #         "description": "A test function",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "param1": {"type": "string", "description": "First parameter"},
    #                 "param2": {"type": "integer", "description": "Second parameter"}
    #             },
    #             "required": ["param1"]
    #         }
    #     }
    # ]

    # providers_to_test = [
    #     ("gemini-2.5-flash", "Gemini"),
    #     ("deepseek-v3p1", "Fireworks"),
    #     ("glm-4-flash", "GLM")
    # ]

    # for model, provider_name in providers_to_test:
    #     try:
    #         formatted = llm.format_function_calls(model, standard_tools)
    #         print(f"✅ {provider_name} formatted tools: {len(formatted)} tools")
    #         if formatted:
    #             first_tool = formatted[0]
    #             if "type" in first_tool:
    #                 print(f"    Format: OpenAI-style (type: {first_tool['type']})")
    #             else:
    #                 print(f"    Format: Direct FunctionDeclaration")
    #     except Exception as e:
    #         print(f"❌ {provider_name} formatting failed: {str(e)[:100]}...")

    # # ========================================
    # # UNIFIED CALL METHOD TESTS
    # # ========================================
    # print("\n" + "=" * 60)
    # print("🔀 TEST 5: UNIFIED CALL METHOD")
    # print("=" * 60)

    # test_cases = [
    #     {
    #         "name": "Gemini Unified Basic",
    #         "model": "gemini-2.5-flash",
    #         "has_tools": False,
    #         "prompt": "Say 'Hello from Gemini!'"
    #     },
    #     {
    #         "name": "GLM Unified Basic",
    #         "model": "glm-4-flash",
    #         "has_tools": False,
    #         "prompt": "Say 'Hello from GLM!'"
    #     },
    #     {
    #         "name": "Fireworks Unified Basic",
    #         "model": "deepseek-v3p1",
    #         "has_tools": False,
    #         "prompt": "Say 'Hello from Fireworks!'"
    #     },
    #     {
    #         "name": "Gemini Unified Tool",
    #         "model": "gemini-2.5-flash",
    #         "has_tools": True,
    #         "prompt": "Get weather for Tokyo"
    #     },
    #     {
    #         "name": "GLM Unified Tool",
    #         "model": "glm-4-flash",
    #         "has_tools": True,
    #         "prompt": "Get weather for Paris"
    #     },
    #     {
    #         "name": "Fireworks Unified Tool",
    #         "model": "deepseek-v3p1",
    #         "has_tools": True,
    #         "prompt": "Calculate 100 + 200"
    #     }
    # ]

    # for test_case in test_cases:
    #     test_num = test_cases.index(test_case) + 1
    #     print(f"\n5.{test_num} Testing {test_case['name']}...")
    #     print(f"  📤 Sending prompt: '{test_case['prompt']}'")
    #     print(f"  🔧 Model: {test_case['model']} | Tools: {'Yes' if test_case['has_tools'] else 'No'}")

    #     # Check if provider is available
    #     if "gemini" in test_case['model'].lower() and not has_gemini:
    #         print("  ⏭️  Skipping (no API key)")
    #         continue
    #     if "deepseek" in test_case['model'].lower() and not has_fireworks:
    #         print("  ⏭️  Skipping (no API key)")
    #         continue
    #     if "glm" in test_case['model'].lower() and not has_glm:
    #         print("  ⏭️  Skipping (no API key)")
    #         continue

    #     if test_case['has_tools']:
    #         print("  📜 Tools being sent:")
    #         print(f"    - {test_tools[0].get('function', {}).get('name', 'Unknown')}")

    #     try:
    #         kwargs = {
    #             "prompt_or_messages": test_case['prompt'],
    #             "model": test_case['model'],
    #             "reasoning_effort": "none"
    #         }

    #         if test_case['has_tools']:
    #             kwargs['tools'] = test_tools[:1]  # Use only weather tool

    #         response = llm.call(**kwargs)

    #         print(f"\n  📥 RAW UNIFIED RESPONSE (Test {test_num}):")
    #         print(f"  ─────────────────────────────")
    #         import json
    #         if isinstance(response, dict):
    #             print(f"  {json.dumps(response, indent=2)}")
    #         else:
    #             print(f"  {response}")
    #         print(f"  ─────────────────────────────")

    #         if isinstance(response, dict):
    #             print(f"  ✅ Response received:")
    #             content = response.get('content', 'No content')
    #             print(f"    Content: {content[:300] if content else content}")
    #             tool_calls = response.get('tool_calls', [])
    #             print(f"    Tool Calls: {len(tool_calls)}")
    #             for i, tc in enumerate(tool_calls):
    #                 func_name = tc.get('function', {}).get('name', 'Unknown')
    #                 func_args = tc.get('function', {}).get('arguments', '{}')
    #                 print(f"      {i+1}. {func_name}({func_args})")
    #         else:
    #             print(f"  ✅ Response: {response[:300]}...")

    #     except Exception as e:
    #         print(f"  ❌ Failed: {str(e)[:300]}...")
    #         import traceback
    #         print(f"  🔍 Traceback: {traceback.format_exc().splitlines()[-1] if traceback.format_exc().splitlines() else 'No traceback'}")

    # Summary
    print("\n" + "=" * 60)
    print("📊 DEMO SUMMARY")
    print("=" * 60)

    print("\n✨ Demo completed! Check the outputs above to verify:")
    print("  • Basic calls work across providers")
    print("  • Tool calling functions correctly")
    print("  • Tool formatting adapts to each provider")
    print("  • Unified call method routes properly")
    print("  • Streaming and reasoning modes function")

    print("\n🔧 Remember to set these environment variables for full testing:")
    print("  • GOOGLE_API_KEY (for Gemini)")
    print("  • FIREWORKS_API_KEY (for Fireworks AI)")
    print("  • GLM_API_KEY or ZHIPU_API_KEY (for GLM/ZhipuAI)")


if __name__ == "__main__":
    _run_demo()
