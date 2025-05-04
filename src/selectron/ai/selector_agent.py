from __future__ import annotations

import traceback
from typing import Any, Callable, Coroutine, Optional, Protocol

from pydantic_ai import Agent, Tool
from pydantic_ai.exceptions import AgentRunError

from selectron.ai.selector_prompt import (
    SELECTOR_PROMPT_BASE,
    SELECTOR_PROMPT_DOM_TEMPLATE,
)
from selectron.ai.selector_tools import SelectorTools
from selectron.ai.types import (
    SelectorProposal,
)
from selectron.util.debug_helpers import save_debug_elements
from selectron.util.logger import get_logger
from selectron.util.model_config import ModelConfig

logger = get_logger(__name__)


# Type alias for the async status callback
StatusCallback = Callable[[str, str, bool], Coroutine[Any, Any, None]]


class ToolStatus(Protocol):
    """Callback protocol for reporting tool execution status."""

    async def __call__(self, *, message: str, state: str, show_spinner: bool) -> None: ...


class Highlighter(Protocol):
    """Callback protocol for interacting with a highlighter mechanism."""

    async def highlight(self, selector: str, color: str) -> bool: ...

    async def clear(self) -> None: ...

    async def show_agent_status(self, text: str, state: str, show_spinner: bool) -> None: ...

    async def hide_agent_status(self) -> None: ...


class SelectorAgentError(Exception):
    """Custom exception for errors during selector agent execution."""

    pass


class SelectorAgent:
    """
    Encapsulates the logic for using an LLM agent to propose CSS selectors based on HTML content.

    This class is designed to be UI-agnostic. It accepts callbacks for status updates
    and highlighting, allowing it to be embedded in different environments (TUI, web, tests).
    """

    # Type alias for the async status callback (duplicated for clarity within class)
    StatusCallback = Callable[[str, str, bool], Coroutine[Any, Any, None]]

    def __init__(
        self,
        *,
        html_content: str,
        dom_string: Optional[str],
        base_url: str,
        model_cfg: ModelConfig,
        status_cb: Optional[StatusCallback] = None,
        highlighter: Optional[Highlighter] = None,
        debug_dump: bool = False,
    ):
        self.html_content = html_content
        self.dom_string = dom_string
        self.base_url = base_url
        self.model_cfg = model_cfg
        self.status_cb = status_cb
        self.highlighter = highlighter
        self.debug_dump = debug_dump

        self._tools_instance = SelectorTools(html_content=self.html_content, base_url=self.base_url)
        self._tool_call_count = 0
        self._best_selector_so_far: Optional[str] = None  # Track the last valid selector found

    async def _safe_status_update(self, message: str, state: str, show_spinner: bool) -> None:
        if self.status_cb:
            try:
                await self.status_cb(message, state, show_spinner)
            except Exception as e:
                logger.error(f"Error in status callback: {e}", exc_info=True)

    async def _safe_highlight(self, selector: str, color: str) -> bool:
        if self.highlighter:
            try:
                return await self.highlighter.highlight(selector, color)
            except Exception as e:
                logger.error(f"Error in highlight callback: {e}", exc_info=True)
                return False
        return False  # Indicate no highlight attempted/successful

    # --- Tool Wrapper Methods ---

    async def _evaluate_selector_wrapper(self, selector: str, target_text_to_check: str, **kwargs):
        self._tool_call_count += 1
        status_prefix = f"Tool #{self._tool_call_count} |"
        await self._safe_status_update(
            f"{status_prefix} evaluate_selector('{selector[:30]}...')",
            state="sending",
            show_spinner=True,
        )

        known_args_for_tool = {
            "anchor_selector": kwargs.get("anchor_selector"),
            "max_html_length": kwargs.get("max_html_length"),
            "max_matches_to_detail": kwargs.get("max_matches_to_detail", None),
            "return_matched_html": True,  # Hardcoded based on previous usage
        }
        filtered_args_for_tool = {k: v for k, v in known_args_for_tool.items() if v is not None}

        result = await self._tools_instance.evaluate_selector(
            selector=selector,
            target_text_to_check=target_text_to_check,
            **filtered_args_for_tool,
        )

        if result and result.element_count > 0 and not result.error:
            await self._safe_highlight(selector, color="yellow")
            await self._safe_status_update(
                f"{status_prefix} evaluate_selector OK ({result.element_count} found)",
                state="received_success",
                show_spinner=True,
            )
            self._best_selector_so_far = selector  # <-- Store successful selector
        elif result and result.element_count == 0 and not result.error:
            await self._safe_status_update(
                f"{status_prefix} Selector found 0 elements",
                state="received_no_results",
                show_spinner=True,
            )
            # Still try to highlight (might clear previous)
            await self._safe_highlight(selector, color="yellow")
        elif result and result.error:
            await self._safe_status_update(
                f"{status_prefix} evaluate_selector Error: {result.error[:50]}...",
                state="received_error",
                show_spinner=True,
            )
        else:  # result is None or unexpected state
            logger.warning(f"_evaluate_selector_wrapper received unexpected result: {result}")
            await self._safe_status_update(
                f"{status_prefix} evaluate_selector unexpected result",
                state="received_error",
                show_spinner=True,
            )

        return result

    async def _get_children_tags_wrapper(self, selector: str, **kwargs):
        self._tool_call_count += 1
        status_prefix = f"[Tool #{self._tool_call_count}]"
        await self._safe_status_update(
            f"{status_prefix} get_children_tags('{selector[:30]}...')",
            state="sending",
            show_spinner=True,
        )
        known_args_for_tool = {
            "anchor_selector": kwargs.get("anchor_selector"),
        }
        filtered_args_for_tool = {k: v for k, v in known_args_for_tool.items() if v is not None}

        result = await self._tools_instance.get_children_tags(
            selector=selector, **filtered_args_for_tool
        )

        if result and result.parent_found and not result.error:
            await self._safe_highlight(selector, color="red")
            await self._safe_status_update(
                f"{status_prefix} get_children_tags OK ({len(result.children_details or [])} children)",
                state="received_success",
                show_spinner=True,
            )
        elif result and not result.parent_found and not result.error:
            await self._safe_status_update(
                f"{status_prefix} Parent selector found 0 elements",
                state="received_no_results",
                show_spinner=True,
            )
            await self._safe_highlight(selector, color="red")  # Highlight parent even if not found?
        elif result and result.error:
            await self._safe_status_update(
                f"{status_prefix} get_children_tags Error: {result.error[:50]}...",
                state="received_error",
                show_spinner=True,
            )
        else:  # result is None or unexpected state
            logger.warning(f"_get_children_tags_wrapper received unexpected result: {result}")
            await self._safe_status_update(
                f"{status_prefix} get_children_tags unexpected result",
                state="received_error",
                show_spinner=True,
            )
        return result

    async def _get_siblings_wrapper(self, selector: str, **kwargs):
        self._tool_call_count += 1
        status_prefix = f"[Tool #{self._tool_call_count}]"
        await self._safe_status_update(
            f"{status_prefix} get_siblings('{selector[:30]}...')",
            state="sending",
            show_spinner=True,
        )
        known_args_for_tool = {
            "anchor_selector": kwargs.get("anchor_selector"),
        }
        filtered_args_for_tool = {k: v for k, v in known_args_for_tool.items() if v is not None}
        result = await self._tools_instance.get_siblings(
            selector=selector, **filtered_args_for_tool
        )

        if result and result.element_found and not result.error:
            await self._safe_highlight(selector, color="blue")
            await self._safe_status_update(
                f"{status_prefix} get_siblings OK ({len(result.siblings or [])} siblings)",
                state="received_success",
                show_spinner=True,
            )
        elif result and not result.element_found and not result.error:
            await self._safe_status_update(
                f"{status_prefix} Element selector found 0 elements",
                state="received_no_results",
                show_spinner=True,
            )
            await self._safe_highlight(
                selector, color="blue"
            )  # Highlight element even if not found?
        elif result and result.error:
            await self._safe_status_update(
                f"{status_prefix} get_siblings Error: {result.error[:50]}...",
                state="received_error",
                show_spinner=True,
            )
        else:  # result is None or unexpected state
            logger.warning(f"_get_siblings_wrapper received unexpected result: {result}")
            await self._safe_status_update(
                f"{status_prefix} get_siblings unexpected result",
                state="received_error",
                show_spinner=True,
            )

        return result

    async def _extract_data_from_element_wrapper(self, selector: str, **kwargs):
        self._tool_call_count += 1
        status_prefix = f"[Tool #{self._tool_call_count}]"
        await self._safe_status_update(
            f"{status_prefix} extract_data_from_element('{selector[:30]}...')",
            state="sending",
            show_spinner=True,
        )
        known_args_for_tool = {
            "attribute_to_extract": kwargs.get("attribute_to_extract"),
            "extract_text": kwargs.get("extract_text", False),  # Default based on previous usage
            "anchor_selector": kwargs.get("anchor_selector"),
        }
        filtered_args_for_tool = {k: v for k, v in known_args_for_tool.items() if v is not None}

        # NOTE: No highlight for extract_data - final highlight happens after run completes
        result = await self._tools_instance.extract_data_from_element(
            selector=selector, **filtered_args_for_tool
        )

        if result and not result.error:
            extracted_count = sum(
                1
                for val in [
                    result.extracted_text,
                    result.extracted_attribute_value,
                    result.extracted_markdown,
                    result.extracted_html,
                ]
                if val is not None
            )
            if extracted_count > 0:
                await self._safe_status_update(
                    f"{status_prefix} extract_data OK ({extracted_count} fields populated)",
                    state="received_success",
                    show_spinner=True,
                )
            else:
                await self._safe_status_update(
                    f"{status_prefix} extract_data OK (No specific data extracted)",
                    state="received_no_results",
                    show_spinner=True,
                )
        elif result and result.error:
            await self._safe_status_update(
                f"{status_prefix} extract_data Error: {result.error[:50]}...",
                state="received_error",
                show_spinner=True,
            )
        else:  # result is None or unexpected state
            logger.warning(
                f"_extract_data_from_element_wrapper received unexpected result: {result}"
            )
            await self._safe_status_update(
                f"{status_prefix} extract_data unexpected result",
                state="received_error",
                show_spinner=True,
            )
        return result

    async def run(self, selector_description: str) -> SelectorProposal:
        """Executes the selector proposal agent workflow."""
        self._tool_call_count = 0  # Reset tool counter for each run
        await self._safe_status_update("Agent starting...", state="thinking", show_spinner=True)
        if not self.html_content:
            logger.error("Cannot run agent: HTML content is missing.")
            await self._safe_status_update(
                "Agent Error: Missing HTML", state="received_error", show_spinner=False
            )
            raise SelectorAgentError("Missing HTML content")

        if not self.base_url:
            logger.error("Cannot run agent: Base URL is missing.")
            await self._safe_status_update(
                "Agent Error: Missing URL", state="received_error", show_spinner=False
            )
            raise SelectorAgentError("Missing base URL")

        try:
            wrapped_tools = [
                Tool(self._evaluate_selector_wrapper),
                Tool(self._get_children_tags_wrapper),
                Tool(self._get_siblings_wrapper),
                Tool(self._extract_data_from_element_wrapper),
            ]

            system_prompt = SELECTOR_PROMPT_BASE
            if self.dom_string:
                system_prompt += SELECTOR_PROMPT_DOM_TEMPLATE.format(
                    dom_representation=self.dom_string
                )
            else:
                logger.warning("Proceeding without DOM string representation.")

            await self._safe_status_update("Thinking...", state="thinking", show_spinner=True)

            agent = Agent(
                self.model_cfg.selector_model,
                output_type=SelectorProposal,
                tools=wrapped_tools,
                system_prompt=system_prompt,
            )

            query_parts = [
                f"Generate the most STABLE CSS selector to target '{selector_description}'.",
                "Prioritize stable attributes and classes.",
                "CRITICAL: Your FINAL output MUST be a single JSON object conforming EXACTLY to the SelectorProposal schema. "
                "This JSON object MUST include values for the fields: 'proposed_selector' (string), 'reasoning' (string), and 'target_cardinality' ('unique' or 'multiple'). "
                "DO NOT include other fields like 'final_verification' or 'extraction_result' in the final JSON output.",
            ]
            query = " ".join(query_parts)
            agent_input: Any = query

            agent_run_result = await agent.run(agent_input)

            if isinstance(agent_run_result.output, SelectorProposal):
                proposal = agent_run_result.output
                logger.info(
                    f"Agent finished. Proposal: {proposal.proposed_selector} (Cardinality: {proposal.target_cardinality})\nREASONING: {proposal.reasoning}"
                )
                # Final success status update is handled by the caller
                # Final highlight is handled by the caller
                # Optional debug dump
                if self.debug_dump:
                    try:
                        await save_debug_elements(
                            tools_instance=self._tools_instance,
                            selector=proposal.proposed_selector,
                            selector_description=selector_description,
                            url=self.base_url,
                            reasoning=proposal.reasoning,
                        )
                    except Exception as dump_err:
                        logger.error(f"Failed to save debug elements: {dump_err}", exc_info=True)

                return proposal
            else:
                logger.error(
                    f"Agent returned unexpected output type: {type(agent_run_result.output)}. Full result: {agent_run_result}"
                )
                await self._safe_status_update(
                    "Agent Error: Unexpected output type",
                    state="received_error",
                    show_spinner=False,
                )
                raise SelectorAgentError(
                    f"Agent returned unexpected output type: {type(agent_run_result.output)}"
                )

        except AgentRunError as agent_err:
            logger.error(f"AgentRunError during agent execution: {agent_err}", exc_info=True)
            await self._safe_status_update(
                f"Agent Error: {type(agent_err).__name__}",
                state="received_error",
                show_spinner=False,
            )
            raise SelectorAgentError(f"Agent run error: {agent_err}") from agent_err
        except Exception as e:
            logger.error(
                f"Unexpected error running SelectorAgent for target '{selector_description}': {e}",
                exc_info=True,
            )
            await self._safe_status_update(
                f"Agent Error: {type(e).__name__}", state="received_error", show_spinner=False
            )
            # Capture traceback for better debugging if needed
            tb_str = traceback.format_exc()
            logger.debug(f"Traceback: {tb_str}")
            raise SelectorAgentError(f"Unexpected agent error: {e}") from e
