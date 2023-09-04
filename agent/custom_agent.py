from pydantic import Field
from langchain.agents import ZeroShotAgent
from langchain.agents.agent import AgentOutputParser


from .custom_parser import CustomMRKLOutputParser2


class CustomZeroShotAgent(ZeroShotAgent):
    output_parser: AgentOutputParser = Field(default_factory=CustomMRKLOutputParser2)

    @classmethod
    def _get_default_output_parser(cls, **kwargs) -> AgentOutputParser:
        return CustomMRKLOutputParser2()

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "ASSISTANT Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "ASSISTANT Thought:"