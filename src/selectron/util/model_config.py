import os
from typing import Literal, Optional

Provider = Literal["anthropic", "openai"]

CLAUDE_3_5_SONNET = "anthropic:claude-3-5-sonnet-latest"
CLAUDE_3_7_SONNET = "anthropic:claude-3-7-sonnet-latest"
OPENAI_GPT_4_1_MINI = "openai:gpt-4.1-mini"
OPENAI_GPT_4_1_NANO = "openai:gpt-4.1-nano"
OPENAI_GPT_4_1 = "openai:gpt-4.1"
ANTHROPIC_ANALYZE_MODEL = CLAUDE_3_5_SONNET
ANTHROPIC_AGENT_MODEL = CLAUDE_3_7_SONNET
ANTHROPIC_CODEGEN_MODEL = CLAUDE_3_7_SONNET
OPENAI_ANALYZE_MODEL = OPENAI_GPT_4_1_NANO
OPENAI_AGENT_MODEL = OPENAI_GPT_4_1
OPENAI_CODEGEN_MODEL = OPENAI_GPT_4_1_MINI


class ModelConfig:
    """Manages LLM provider configuration based on available API keys."""

    anthropic_key: Optional[str]
    openai_key: Optional[str]
    provider: Provider
    _propose_model: str
    _agent_model: str

    def __init__(self):
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        maybe_provider = self._determine_provider()
        if not maybe_provider:
            raise ValueError(
                "Missing required environment variable: Set ANTHROPIC_API_KEY or OPENAI_API_KEY"
            )
        self.provider = maybe_provider
        # Set models directly based on provider
        if self.provider == "anthropic":
            self._propose_model = ANTHROPIC_ANALYZE_MODEL
            self._agent_model = ANTHROPIC_AGENT_MODEL
            self._codegen_model = ANTHROPIC_CODEGEN_MODEL
        elif self.provider == "openai":
            self._propose_model = OPENAI_ANALYZE_MODEL
            self._agent_model = OPENAI_AGENT_MODEL
            self._codegen_model = OPENAI_CODEGEN_MODEL

    def _determine_provider(self) -> Optional[Provider]:
        if self.anthropic_key:
            return "anthropic"
        elif self.openai_key:
            return "openai"
        else:
            return None

    @property
    def analyze_model(self) -> str:
        return self._propose_model

    @property
    def agent_model(self) -> str:
        return self._agent_model

    @property
    def codegen_model(self) -> str:
        return self._codegen_model

    @property
    def api_key(self) -> str:
        """Returns the API key for the selected provider."""
        key = None
        if self.provider == "anthropic":
            key = self.anthropic_key
        elif self.provider == "openai":
            key = self.openai_key
        if key is None:
            raise ValueError(
                f"API key for provider '{self.provider}' not found, though provider was set."
            )
        return key
