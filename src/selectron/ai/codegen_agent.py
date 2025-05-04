from __future__ import annotations

import json
import typing
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext

from selectron.ai.codegen_prompt import CODEGEN_PROMPT
from selectron.ai.codegen_utils import (
    clean_agent_code,
    validate_cross_key_duplicates,
    validate_empty_columns,
    validate_identical_columns,
    validate_internal_repetition,
    validate_naive_text_match,
    validate_redundant_key_pairs,
    validate_result,
    validate_text_representation,
)
from selectron.util.get_app_dir import get_app_dir
from selectron.util.logger import get_logger
from selectron.util.model_config import ModelConfig
from selectron.util.sample_items import sample_items
from selectron.util.slugify_url import slugify_url

logger = get_logger(__name__)


class CodegenAgent:
    class _CodeEvaluationResult(BaseModel):
        success: bool = Field(
            ..., description="Whether the code executed successfully and passed validation."
        )
        feedback: str = Field(
            ...,
            description="Error message if success is false, or 'success' if true (may include quality feedback).",
        )
        sampled_output_with_html: Optional[str] = Field(
            default=None,
            description="JSON string of ONE sampled input/output pair: {html_input: str, extracted_data: dict} if successful, otherwise null.",
        )
        iteration_count: int = Field(
            ..., description="The current evaluation iteration number (starting from 1)."
        )

    # Type alias for the async status callback
    StatusCallback = Callable[[str, str, bool], Coroutine[Any, Any, None]]

    def __init__(
        self,
        *,
        html_samples: List[str],
        model_cfg: Optional[ModelConfig] = None,
        save_results: bool = False,
        base_url: Optional[str] = None,
        input_selector: Optional[str] = None,
        input_selector_description: Optional[str] = None,
        status_cb: Optional[StatusCallback] = None,  # <-- Add status callback
    ):
        if not html_samples:
            raise ValueError("html_samples must be a non-empty list")
        if save_results and not base_url:
            raise ValueError("base_url must be provided if save_results is True")

        self.html_samples = html_samples
        self.model_cfg = model_cfg or ModelConfig()
        self.save_results = save_results
        self.base_url = base_url
        self.input_selector = input_selector
        self.input_selector_description = input_selector_description
        self.status_cb = status_cb  # <-- Store status callback

        # Determine the correct directory to save parsers
        self.parser_save_dir: Optional[Path] = None
        if self.save_results:
            src_parsers_dir = Path("src/selectron/parsers")
            if src_parsers_dir.is_dir():
                self.parser_save_dir = src_parsers_dir
                logger.info(f"Saving to source directory: {self.parser_save_dir}")
            else:
                app_parsers_dir = get_app_dir() / "parsers"
                self.parser_save_dir = app_parsers_dir
                logger.info(
                    f"Source parser directory not found. Using app config directory for saving: {self.parser_save_dir}"
                )

            if self.parser_save_dir:
                try:
                    self.parser_save_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.error(
                        f"Failed to create parser save directory {self.parser_save_dir}: {e}",
                        exc_info=True,
                    )
                    # Decide if we should raise an error or just warn and disable saving
                    logger.warning("Disabling parser saving due to directory creation error.")
                    self.save_results = False  # Disable saving if dir creation fails
                    self.parser_save_dir = None

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _exec_candidate(
        self, code: str, html_samples_inner: List[str]
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """Execute candidate code in a sandbox and run parse_element over samples."""
        sandbox: Dict[str, Any] = {"BeautifulSoup": BeautifulSoup, "json": json}
        try:
            compiled = compile(code, "<agent_code>", "exec")
        except SyntaxError as e:
            feedback = f"syntax error: {e.msg} (line {e.lineno})"
            logger.warning(feedback)
            return False, feedback, []
        try:
            exec(compiled, sandbox)
        except Exception as e:  # Granular exceptions handled by sandbox code itself
            feedback = f"runtime error during exec: {type(e).__name__}: {e}"
            logger.warning(feedback, exc_info=True)
            return False, feedback, []
        parse_fn = sandbox.get("parse_element")
        if not callable(parse_fn):
            feedback = "function `parse_element(html: str) -> dict` not found"
            logger.warning(feedback)
            return False, feedback, []

        outputs: List[Dict[str, Any]] = []
        for idx, html in enumerate(html_samples_inner):
            try:
                result = parse_fn(html)
            except Exception as e:  # pragma: no cover – we still capture & validate
                feedback = (
                    f"error when calling parse_element on sample {idx}: {type(e).__name__}: {e}"
                )
                logger.warning(feedback, exc_info=True)
                return False, feedback, []
            ok, msg = validate_result(result)
            if not ok:
                feedback = f"invalid return value for sample {idx}: {msg}"
                logger.warning(feedback)
                return False, feedback, []
            # Ensure pyright understands the type after validation
            outputs.append(typing.cast(Dict[str, Any], result))
        return True, "success", outputs

    async def run(self) -> Tuple[str, List[Dict[str, Any]]]:
        agent = Agent(
            self.model_cfg.codegen_model,
            system_prompt=CODEGEN_PROMPT,
        )

        @agent.tool(retries=3)
        async def evaluate_and_sample_code(
            ctx: RunContext[None], code: str, iteration_count: int
        ) -> "CodegenAgent._CodeEvaluationResult":
            logger.info(
                f"Tool evaluate_and_sample_code called with iteration_count: {iteration_count}"
            )
            if self.status_cb:
                await self.status_cb(
                    f"Evaluating code (Iteration {iteration_count})...", "thinking", True
                )
            cleaned_code = clean_agent_code(code)

            success, feedback, outputs = self._exec_candidate(cleaned_code, self.html_samples)

            if success:
                logger.info("Agent code passed validation")
                quality_feedback: List[str] = []
                if len(outputs) > 1:
                    all_keys = set().union(*(d.keys() for d in outputs))
                    quality_feedback.extend(validate_empty_columns(outputs, all_keys))
                    quality_feedback.extend(validate_identical_columns(outputs, all_keys))
                    quality_feedback.extend(
                        validate_text_representation(outputs, self.html_samples)
                    )
                    quality_feedback.extend(validate_redundant_key_pairs(outputs, all_keys))
                    quality_feedback.extend(validate_cross_key_duplicates(outputs, all_keys))
                    quality_feedback.extend(validate_internal_repetition(outputs, all_keys))
                    quality_feedback.extend(validate_naive_text_match(outputs, self.html_samples))

                quality_feedback_str = "\n- ".join(quality_feedback) if quality_feedback else "None"
                if quality_feedback:
                    feedback = f"Quality issues detected:\n- {quality_feedback_str}"
                else:
                    feedback = (
                        "Code executed successfully with no quality issues detected by validation."
                    )

                sampled_outputs_with_indices = sample_items(outputs, sample_size=1)
                sampled_output_str: Optional[str] = None
                if sampled_outputs_with_indices:
                    original_index, sample_dict = sampled_outputs_with_indices[0]
                    paired_sample = {
                        "html_input": self.html_samples[original_index],
                        "extracted_data": sample_dict,
                    }
                    sampled_output_str = json.dumps(paired_sample, indent=2, ensure_ascii=False)

                if iteration_count == 1:
                    retry_message = (
                        "MANDATORY REFINEMENT (Iteration 1 succeeded, but iteration 2 is required):\n"
                        "Analyze the quality feedback and the sample output provided below.\n"
                        "Refine your code to address any issues (e.g., missing data, redundancy, exhaustiveness) and call the tool again.\n\n"
                        f"Quality Feedback:\n{feedback}\n\n"
                        f"Sample Input/Output Pair:\n{sampled_output_str}"
                    )
                    logger.info("Raising ModelRetry: Forcing mandatory iteration 2 due to policy…")
                    if self.status_cb:
                        await self.status_cb(
                            "Refining code (Mandatory Iteration 2)...", "thinking", True
                        )
                    raise ModelRetry(retry_message)
            else:
                logger.warning(f"Raising ModelRetry: Code validation failed: {feedback}")
                raise ModelRetry(feedback)

            return CodegenAgent._CodeEvaluationResult(
                success=success,
                feedback=feedback,
                sampled_output_with_html=sampled_output_str,
                iteration_count=iteration_count,
            )

        initial_iteration = 1
        response = await agent.run(
            f"generate the initial python code and evaluate it using the tool (iteration {initial_iteration})."
        )
        final_code_obj = response.output
        logger.info("Agent run finished.")
        final_code = clean_agent_code(final_code_obj)
        success, feedback, final_outputs = self._exec_candidate(final_code, self.html_samples)
        if not success:
            logger.error(
                f"INTERNAL ERROR: Agent returned code that failed final validation: {feedback}"
            )
            raise RuntimeError(
                f"agent returned non-working code despite internal validation: {feedback}"
            )

        # --- Save results if configured ---
        if self.save_results and self.parser_save_dir:
            assert self.base_url is not None  # Ensured by __init__ validation

            # Determine if we are saving to the user directory or the source directory
            app_parsers_dir = get_app_dir() / "parsers"
            is_saving_to_user_dir = self.parser_save_dir == app_parsers_dir

            url_slug = slugify_url(self.base_url)
            save_path = self.parser_save_dir / f"{url_slug}.json"

            result_data = {
                "python": final_code,
                "selector": self.input_selector or "",
                "selector_description": self.input_selector_description or "",
            }

            try:
                if is_saving_to_user_dir:
                    # Allow overwriting in the user directory
                    was_existing = save_path.exists()
                    save_path.write_text(json.dumps(result_data, indent=2, ensure_ascii=False))
                    if was_existing:
                        logger.info(f"overwrote existing parser in user directory at {save_path}")
                    else:
                        logger.info(f"saved new parser to user directory at {save_path}")
                else:
                    # Prevent overwriting in the source directory
                    if not save_path.exists():
                        save_path.write_text(json.dumps(result_data, indent=2, ensure_ascii=False))
                        logger.info(f"saved new codegen result to source directory {save_path}")
                    else:
                        logger.warning(
                            f"skipped saving to source directory: file already exists at {save_path}"
                        )
            except Exception as e:
                logger.error(f"failed to write result to {save_path}: {e}", exc_info=True)

        elif self.save_results and not self.parser_save_dir:
            logger.warning(
                "Skipped saving: parser save directory was not correctly initialized or created."
            )

        return final_code, final_outputs
