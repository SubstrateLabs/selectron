import os
from typing import Literal, Optional

Provider = Literal["anthropic", "openai"]

CLAUDE_3_5_SONNET = "anthropic:claude-3-5-sonnet-latest"
CLAUDE_3_7_SONNET = "anthropic:claude-3-7-sonnet-latest"
OPENAI_GPT_4_1_MINI = "openai:gpt-4.1-mini"
OPENAI_GPT_4_1_NANO = "openai:gpt-4.1-nano"
OPENAI_GPT_4_1 = "openai:gpt-4.1"
ANTHROPIC_ANALYZE_MODEL = CLAUDE_3_5_SONNET
ANTHROPIC_SELECTOR_MODEL = CLAUDE_3_7_SONNET
ANTHROPIC_CODEGEN_MODEL = CLAUDE_3_7_SONNET
OPENAI_ANALYZE_MODEL = OPENAI_GPT_4_1_MINI
OPENAI_SELECTOR_MODEL = OPENAI_GPT_4_1
OPENAI_CODEGEN_MODEL = OPENAI_GPT_4_1


class ModelConfig:
    """Manages LLM provider configuration based on available API keys."""

    anthropic_key: Optional[str]
    openai_key: Optional[str]
    provider: Optional[Provider]
    _propose_model: Optional[str] = None
    _selector_model: Optional[str] = None
    _codegen_model: Optional[str] = None

    def __init__(self):
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        self.provider = self._determine_provider()

        # Set models only if a provider is available
        if self.provider == "anthropic":
            self._propose_model = ANTHROPIC_ANALYZE_MODEL
            self._selector_model = ANTHROPIC_SELECTOR_MODEL
            self._codegen_model = ANTHROPIC_CODEGEN_MODEL
        elif self.provider == "openai":
            self._propose_model = OPENAI_ANALYZE_MODEL
            self._selector_model = OPENAI_SELECTOR_MODEL
            self._codegen_model = OPENAI_CODEGEN_MODEL
        # No else needed, models remain None if no provider

    def _determine_provider(self) -> Optional[Provider]:
        if self.anthropic_key:
            return "anthropic"
        elif self.openai_key:
            return "openai"
        else:
            return None

    @property
    def analyze_model(self) -> str:
        if self._propose_model is None:
            raise ValueError("AI provider not configured, cannot get analyze model.")
        return self._propose_model

    @property
    def selector_model(self) -> str:
        if self._selector_model is None:
            raise ValueError("AI provider not configured, cannot get selector model.")
        return self._selector_model

    @property
    def codegen_model(self) -> str:
        if self._codegen_model is None:
            raise ValueError("AI provider not configured, cannot get codegen model.")
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
            # This case should ideally not be reached if provider is None,
            # but added for robustness / if provider is somehow set without a key.
            # Or, more likely, if code tries to access api_key when provider is None.
            raise ValueError(
                f"API key for provider '{self.provider}' not found or provider not configured."
            )
        return key
