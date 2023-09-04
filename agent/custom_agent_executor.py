from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from langchain.tools.base import BaseTool
from langchain.agents import AgentExecutor
from langchain.schema import AgentAction, AgentFinish

from .tools import CustomInvalidTool

class CustomAgentExecutor(AgentExecutor):
    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        # Call the LLM to see what to do.
        output = self.agent.plan(intermediate_steps, **inputs)
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        result = []
        for agent_action in actions:
            self.callback_manager.on_agent_action(
                agent_action, verbose=self.verbose, color="green"
            )
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                # =============================== modify ===============================
                # give GetDetailsTool more kwargs
                tool_run_kwargs["inputs"] = inputs
                # =============================== modify ===============================
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                # We then call the tool on the tool input to get an observation
                observation = tool.run(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    **tool_run_kwargs,
                )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = CustomInvalidTool().run(
                    agent_action.tool,
                    all_tools = list(name_to_tool_map.keys()),
                    verbose=self.verbose,
                    color=None,
                    **tool_run_kwargs,
                )
            result.append((agent_action, observation))
        return result