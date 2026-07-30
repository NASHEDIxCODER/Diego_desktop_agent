"""
Agent Mode — Desktop AI Agent for Leo.

Transforms Leo from a command-based assistant into a production-grade
desktop AI agent capable of:
- Seeing the screen
- Understanding UI
- Controlling browser with existing profile
- Using mouse and keyboard
- Planning and executing multi-step tasks
- Recovering from failures

Architecture:
  Voice → AgentPlanner → AgentExecutor → Browser/Desktop
                ↑              ↑
            ScreenVision    AgentMemory
"""

from agent.planner import AgentPlanner, agent_planner
from agent.executor import AgentExecutor, agent_executor
from agent.memory import AgentMemory, agent_memory
from agent.browser import BrowserController, browser_controller

__all__ = [
    "AgentPlanner", "agent_planner",
    "AgentExecutor", "agent_executor",
    "AgentMemory", "agent_memory",
    "BrowserController", "browser_controller",
]