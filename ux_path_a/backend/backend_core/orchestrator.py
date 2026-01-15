"""
LLM Orchestrator for UX Path A.

This module handles the orchestration of LLM calls, tool selection,
and response generation. It enforces platform invariants and ensures
all analytics come from tools.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
from openai import OpenAI
from openai import APIError

# Use absolute imports (works in both local and Railway with PYTHONPATH=/app)
from ux_path_a.backend.backend_core.config import settings
from ux_path_a.backend.backend_core.tools.registry import ToolRegistry
from ux_path_a.backend.backend_core.prompts import get_system_prompt

# Try to import centralized LLM config utilities
try:
    import sys
    from pathlib import Path
    project_root = settings.PLATFORM_ROOT
    sys.path.insert(0, str(project_root))
    from core.utils.llm_config import get_llm_models, is_newer_model, get_model_capabilities
    _has_llm_config = True
except Exception:
    _has_llm_config = False

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """
    Orchestrates chat interactions with LLM and tools.
    
    Responsibilities:
    - Manage LLM calls
    - Route tool calls
    - Enforce invariants
    - Track token usage
    - Maintain conversation context
    """
    
    def __init__(self):
        """Initialize orchestrator."""
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.tool_registry = ToolRegistry()
        self.system_prompt = get_system_prompt()
        
        # Register tools
        self._register_tools()
    
    def _register_tools(self):
        """Register all available tools."""
        # Try absolute import first (for local development), fallback to relative (for deployment)
        # Use absolute imports (works in both local and Railway with PYTHONPATH=/app)
        from ux_path_a.backend.backend_core.tools.data_tools import register_data_tools
        from ux_path_a.backend.backend_core.tools.analysis_tools import register_analysis_tools
        from ux_path_a.backend.backend_core.tools.web_search_tools import register_web_search_tools
        
        # Register all tool categories
        register_data_tools(self.tool_registry)
        register_analysis_tools(self.tool_registry)
        register_web_search_tools(self.tool_registry)
        
        logger.info(f"Registered {len(self.tool_registry.get_function_definitions())} tools")
    
    async def process_message(
        self,
        message: str,
        session_id: str,
        conversation_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Process a user message and generate response.
        
        Args:
            message: User message text
            session_id: Session identifier
            conversation_history: Previous messages in conversation
            
        Returns:
            Response dictionary with content, tool_calls, token_usage
        """
        # Build messages for LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        # Add conversation history
        for msg in conversation_history[-10:]:  # Last 10 messages for context
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Get tool definitions for function calling
        tools = self.tool_registry.get_function_definitions()
        
        try:
            # Call LLM with function calling
            # Handle different parameter requirements for different models
            # Use centralized config if available, otherwise fallback to name-based detection
            if _has_llm_config:
                is_newer_model_flag = is_newer_model(settings.OPENAI_MODEL, project_root=settings.PLATFORM_ROOT)
            else:
                is_newer_model_flag = "gpt-5" in settings.OPENAI_MODEL.lower() or "o1" in settings.OPENAI_MODEL.lower()
            
            llm_params = {
                "model": settings.OPENAI_MODEL,
                "messages": messages,
            }
            
            # Temperature is not supported by all models (e.g., gpt-5-mini, o1)
            if not is_newer_model_flag:
                llm_params["temperature"] = settings.OPENAI_TEMPERATURE
            
            # Add function calling if tools available
            # Newer models (gpt-5, o1) use "tools" parameter, older models use "functions"
            if tools:
                if is_newer_model_flag:
                    # Convert to tools format for newer models
                    llm_params["tools"] = [{"type": "function", "function": tool} for tool in tools]
                    llm_params["tool_choice"] = "auto"
                else:
                    # Legacy function calling format
                    llm_params["functions"] = tools
                    llm_params["function_call"] = "auto"
            
            # Use appropriate parameter based on model
            # For newer models (gpt-5, o1): try max_completion_tokens, but SDK might not support it
            # For older models: use max_tokens
            # Handle both SDK-level errors (TypeError) and API-level errors (APIError)
            if is_newer_model_flag:
                llm_params["max_completion_tokens"] = settings.OPENAI_MAX_TOKENS
            else:
                llm_params["max_tokens"] = settings.OPENAI_MAX_TOKENS
            
            try:
                response = self.client.chat.completions.create(**llm_params)
            except TypeError as e:
                # SDK-level error: parameter not recognized by SDK
                error_str = str(e).lower()
                if "max_completion_tokens" in error_str and "unexpected keyword" in error_str:
                    # SDK doesn't support max_completion_tokens parameter, omit it
                    # The API will use its default or we'll get an API error telling us what to use
                    logger.warning(f"SDK doesn't support max_completion_tokens parameter, omitting token limit...")
                    llm_params.pop("max_completion_tokens", None)
                    response = self.client.chat.completions.create(**llm_params)
                else:
                    raise
            except (APIError, Exception) as e:
                # Check if error is about unsupported parameter
                error_str = str(e).lower()
                error_message = error_str
                
                # Try multiple ways to extract error message from APIError object
                if hasattr(e, 'body') and isinstance(e.body, dict):
                    error_obj = e.body.get('error', {})
                    if isinstance(error_obj, dict):
                        error_message = str(error_obj.get('message', '')).lower()
                elif hasattr(e, 'response') and hasattr(e.response, 'json'):
                    try:
                        error_data = e.response.json()
                        if isinstance(error_data, dict) and 'error' in error_data:
                            error_obj = error_data.get('error', {})
                            if isinstance(error_obj, dict):
                                error_message = str(error_obj.get('message', '')).lower()
                    except:
                        pass
                elif hasattr(e, 'message'):
                    error_message = str(e.message).lower()
                
                # Combine all error sources for checking
                combined_error = f"{error_str} {error_message}".lower()
                
                # Check if max_tokens is not supported and needs max_completion_tokens
                if ("max_tokens" in combined_error) and \
                   ("unsupported" in combined_error) and \
                   ("max_completion_tokens" in combined_error):
                    # API says max_tokens not supported, need max_completion_tokens
                    logger.warning(f"Model {settings.OPENAI_MODEL} requires max_completion_tokens instead of max_tokens, retrying...")
                    llm_params.pop("max_tokens", None)
                    llm_params["max_completion_tokens"] = settings.OPENAI_MAX_TOKENS
                    response = self.client.chat.completions.create(**llm_params)
                # Check if max_completion_tokens is not supported and needs max_tokens
                elif ("max_completion_tokens" in combined_error) and \
                     ("unsupported" in combined_error) and \
                     ("max_tokens" in combined_error):
                    # API says max_completion_tokens not supported, need max_tokens
                    logger.warning(f"Model {settings.OPENAI_MODEL} requires max_tokens instead of max_completion_tokens, retrying...")
                    llm_params.pop("max_completion_tokens", None)
                    llm_params["max_tokens"] = settings.OPENAI_MAX_TOKENS
                    response = self.client.chat.completions.create(**llm_params)
                else:
                    # Some other error, re-raise it
                    raise
            
            message_response = response.choices[0].message
            
            # Extract thinking/reasoning content if available (for models like o1)
            thinking_content = None
            if hasattr(message_response, 'reasoning_content') and message_response.reasoning_content:
                thinking_content = message_response.reasoning_content
            elif hasattr(response.choices[0], 'reasoning_content') and response.choices[0].reasoning_content:
                thinking_content = response.choices[0].reasoning_content
            
            # Handle function calls (check both function_call and tool_calls)
            function_calls = []
            if hasattr(message_response, 'function_call') and message_response.function_call:
                function_calls.append(message_response.function_call)
            elif hasattr(message_response, 'tool_calls') and message_response.tool_calls:
                function_calls = message_response.tool_calls
            
            if function_calls:
                tool_results = await self._execute_tools(
                    function_calls=function_calls,
                    session_id=session_id,
                )
                
                # Get final response with tool results
                assistant_msg = {
                    "role": "assistant",
                    "content": message_response.content or "",
                }
                if hasattr(message_response, 'function_call') and message_response.function_call:
                    assistant_msg["function_call"] = message_response.function_call
                elif hasattr(message_response, 'tool_calls') and message_response.tool_calls:
                    assistant_msg["tool_calls"] = message_response.tool_calls
                messages.append(assistant_msg)
                
                # Add tool results
                # Newer models use "tool" role, older models use "function" role
                for tool_result in tool_results:
                    if is_newer_model_flag:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_result.get("tool_call_id", ""),
                            "content": tool_result["result"],
                        })
                    else:
                        messages.append({
                            "role": "function",
                            "name": tool_result["name"],
                            "content": tool_result["result"],
                        })
                
                # Get final LLM response
                # Use same model type check (is_newer_model_flag already defined above)
                final_llm_params = {
                    "model": settings.OPENAI_MODEL,
                    "messages": messages,
                }
                
                # Temperature is not supported by all models (e.g., gpt-5-mini, o1)
                if not is_newer_model_flag:
                    final_llm_params["temperature"] = settings.OPENAI_TEMPERATURE
                
                # Use appropriate parameter based on model
                # For newer models (gpt-5, o1): try max_completion_tokens, but SDK might not support it
                # For older models: use max_tokens
                # Handle both SDK-level errors (TypeError) and API-level errors (APIError)
                if is_newer_model_flag:
                    final_llm_params["max_completion_tokens"] = settings.OPENAI_MAX_TOKENS
                else:
                    final_llm_params["max_tokens"] = settings.OPENAI_MAX_TOKENS
                
                try:
                    final_response = self.client.chat.completions.create(**final_llm_params)
                except TypeError as e:
                    # SDK-level error: parameter not recognized by SDK
                    error_str = str(e).lower()
                    if "max_completion_tokens" in error_str and "unexpected keyword" in error_str:
                        # SDK doesn't support max_completion_tokens parameter, omit it
                        # The API will use its default or we'll get an API error telling us what to use
                        logger.warning(f"SDK doesn't support max_completion_tokens parameter, omitting token limit...")
                        final_llm_params.pop("max_completion_tokens", None)
                        final_response = self.client.chat.completions.create(**final_llm_params)
                    else:
                        raise
                except (APIError, Exception) as e:
                    # Check if error is about unsupported parameter
                    error_str = str(e).lower()
                    error_message = error_str
                    
                    # Try multiple ways to extract error message from APIError object
                    if hasattr(e, 'body') and isinstance(e.body, dict):
                        error_obj = e.body.get('error', {})
                        if isinstance(error_obj, dict):
                            error_message = str(error_obj.get('message', '')).lower()
                    elif hasattr(e, 'response') and hasattr(e.response, 'json'):
                        try:
                            error_data = e.response.json()
                            if isinstance(error_data, dict) and 'error' in error_data:
                                error_obj = error_data.get('error', {})
                                if isinstance(error_obj, dict):
                                    error_message = str(error_obj.get('message', '')).lower()
                        except:
                            pass
                    elif hasattr(e, 'message'):
                        error_message = str(e.message).lower()
                    
                    # Combine all error sources for checking
                    combined_error = f"{error_str} {error_message}".lower()
                    
                    # Check if max_tokens is not supported and needs max_completion_tokens
                    if ("max_tokens" in combined_error) and \
                       ("unsupported" in combined_error) and \
                       ("max_completion_tokens" in combined_error):
                        # API says max_tokens not supported, need max_completion_tokens
                        logger.warning(f"Model {settings.OPENAI_MODEL} requires max_completion_tokens instead of max_tokens, retrying...")
                        final_llm_params.pop("max_tokens", None)
                        final_llm_params["max_completion_tokens"] = settings.OPENAI_MAX_TOKENS
                        final_response = self.client.chat.completions.create(**final_llm_params)
                    # Check if max_completion_tokens is not supported and needs max_tokens
                    elif ("max_completion_tokens" in combined_error) and \
                         ("unsupported" in combined_error) and \
                         ("max_tokens" in combined_error):
                        # API says max_completion_tokens not supported, need max_tokens
                        logger.warning(f"Model {settings.OPENAI_MODEL} requires max_tokens instead of max_completion_tokens, retrying...")
                        final_llm_params.pop("max_completion_tokens", None)
                        final_llm_params["max_tokens"] = settings.OPENAI_MAX_TOKENS
                        final_response = self.client.chat.completions.create(**final_llm_params)
                    else:
                        # Some other error, re-raise it
                        raise
                
                final_message_response = final_response.choices[0].message
                content = final_message_response.content
                
                # Extract thinking/reasoning content from final response if available
                if not thinking_content:
                    if hasattr(final_message_response, 'reasoning_content') and final_message_response.reasoning_content:
                        thinking_content = final_message_response.reasoning_content
                    elif hasattr(final_response.choices[0], 'reasoning_content') and final_response.choices[0].reasoning_content:
                        thinking_content = final_response.choices[0].reasoning_content
                
                # Combine token usage from both calls
                total_prompt_tokens = response.usage.prompt_tokens + final_response.usage.prompt_tokens
                total_completion_tokens = response.usage.completion_tokens + final_response.usage.completion_tokens
                total_tokens = response.usage.total_tokens + final_response.usage.total_tokens
            else:
                content = message_response.content
                tool_results = None
                total_prompt_tokens = response.usage.prompt_tokens
                total_completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
            
            # Serialize tool_calls to ensure they're JSON serializable
            tool_calls_serialized = None
            if function_calls:
                try:
                    import json
                    def serialize_tool_call(tc):
                        """Serialize a tool call object to a dict, handling nested Function objects."""
                        # Check if it's a Function-like object (has name and arguments attributes)
                        # This check must come before checking __dict__ because Function might not expose __dict__
                        try:
                            if hasattr(tc, 'name') and hasattr(tc, 'arguments'):
                                # This looks like a Function object from OpenAI SDK
                                func_dict = {
                                    'name': getattr(tc, 'name', None),
                                    'arguments': getattr(tc, 'arguments', None),
                                }
                                # Add optional attributes
                                if hasattr(tc, 'id'):
                                    func_dict['id'] = getattr(tc, 'id', None)
                                if hasattr(tc, 'type'):
                                    func_dict['type'] = getattr(tc, 'type', None)
                                return func_dict
                        except:
                            pass
                        
                        # Handle callable objects (functions, methods)
                        if callable(tc) and not isinstance(tc, type):
                            return f"<function: {getattr(tc, '__name__', str(tc))}>"
                        
                        if isinstance(tc, dict):
                            # Recursively serialize dict values
                            return {k: serialize_tool_call(v) for k, v in tc.items()}
                        elif hasattr(tc, '__dict__'):
                            result = {}
                            for k, v in tc.__dict__.items():
                                if not k.startswith('_'):
                                    result[k] = serialize_tool_call(v)
                            # Also check for attributes that might not be in __dict__ (like properties)
                            for attr in ['name', 'arguments', 'id', 'type', 'function']:
                                if hasattr(tc, attr) and attr not in result:
                                    try:
                                        attr_value = getattr(tc, attr)
                                        result[attr] = serialize_tool_call(attr_value)
                                    except:
                                        pass
                            return result
                        elif isinstance(tc, (str, int, float, bool, type(None))):
                            return tc
                        elif isinstance(tc, (list, tuple)):
                            return [serialize_tool_call(item) for item in tc]
                        else:
                            return str(tc)
                    
                    tool_calls_serialized = [serialize_tool_call(tc) for tc in function_calls]
                    # Test serialization
                    json.dumps(tool_calls_serialized)
                except Exception as e:
                    logger.warning(f"Could not serialize tool_calls in orchestrator: {e}", exc_info=True)
                    # Fallback: convert everything to string representation
                    try:
                        tool_calls_serialized = [str(tc) for tc in function_calls] if function_calls else None
                    except:
                        tool_calls_serialized = None
            
            return {
                "content": content,
                "thinking_content": thinking_content,  # Reasoning/thinking process from models like o1
                "tool_calls": tool_calls_serialized if function_calls else None,
                "tool_results": tool_results,
                "token_usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_tokens,
                },
            }
            
        except Exception as e:
            logger.error(f"Error in LLM call: {e}", exc_info=True)
            raise
    
    async def _execute_tools(
        self,
        function_calls: List[Any],
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Execute tool calls and return results.
        
        Args:
            function_calls: List of function call objects from LLM
            session_id: Session identifier for audit logging
            
        Returns:
            List of tool execution results
        """
        results = []
        
        for func_call in function_calls:
            # Handle both function_call and tool_call formats
            tool_call_id = None
            if hasattr(func_call, 'function'):
                # Newer tool_calls format
                tool_name = func_call.function.name
                tool_args_str = func_call.function.arguments
                if hasattr(func_call, 'id'):
                    tool_call_id = func_call.id
            elif hasattr(func_call, 'id'):
                # Tool call with id
                tool_call_id = func_call.id
                if hasattr(func_call, 'function'):
                    tool_name = func_call.function.name
                    tool_args_str = func_call.function.arguments
                else:
                    tool_name = func_call.name if hasattr(func_call, 'name') else str(func_call)
                    tool_args_str = func_call.arguments if hasattr(func_call, 'arguments') else "{}"
            else:
                # Older function_call format
                tool_name = func_call.name
                tool_args_str = func_call.arguments
            
            # Parse arguments
            import json
            try:
                tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
            except:
                tool_args = eval(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
            
            logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
            
            try:
                # Execute tool via registry
                result = await self.tool_registry.execute_tool(
                    tool_name=tool_name,
                    arguments=tool_args,
                    session_id=session_id,
                )
                
                results.append({
                    "name": tool_name,
                    "result": str(result),
                    "tool_call_id": tool_call_id,
                })
                
                # TODO: Audit logging (INV-AUDIT-01, INV-AUDIT-02)
                
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
                results.append({
                    "name": tool_name,
                    "result": f"Error: {str(e)}",
                    "tool_call_id": tool_call_id,
                })
        
        return results
