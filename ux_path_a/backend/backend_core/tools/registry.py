"""
Tool Registry for Path A.

Manages registration and execution of tools that connect to the
Trading Copilot Platform. All tools must return data from the platform,
never fabricate values (INV-LLM-01, INV-LLM-02).
"""

from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class Tool(ABC):
    """Base class for all tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Tool parameters schema (OpenAI function calling format)."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        pass


class ToolRegistry:
    """
    Registry for managing and executing tools.
    
    All tools must connect to the Trading Copilot Platform
    and return real data (INV-LLM-02).
    """
    
    def __init__(self):
        """Initialize tool registry."""
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: str,
    ) -> Any:
        """
        Execute a tool with given arguments.
        
        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments
            session_id: Session ID for audit logging
            
        Returns:
            Tool execution result
        """
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        
        # TODO: Audit logging (INV-AUDIT-01)
        logger.info(f"Executing tool: {tool_name} for session: {session_id}")
        
        return await tool.execute(**arguments)
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI function calling format definitions for all tools.
        
        Returns:
            List of function definitions in OpenAI format
        """
        definitions = []
        for tool in self._tools.values():
            definitions.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })
        return definitions
