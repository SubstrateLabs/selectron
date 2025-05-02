import asyncio
import os

from selectron.ai.selector_agent import AgentResult, SelectorAgent
from selectron.util.logger import get_logger

logger = get_logger(__name__)

# More complex HTML with unstable identifiers
DUMMY_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Agent Stability Test Page</title>
    <style>
        .item { border: 1px solid #eee; padding: 5px; margin: 5px; }
        .highlight { background-color: yellow; }
        .data-point { font-style: italic; }
        .global-nav__primary-link { text-decoration: none; margin: 0 10px; color: #333; }
        .global-nav__primary-link--active { font-weight: bold; }
        .notification-badge { background-color: red; color: white; border-radius: 50%; padding: 1px 5px; font-size: 0.8em; margin-left: 4px; }
        .notification-badge--show { display: inline-block; }
        .visually-hidden { position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0; overflow: hidden; clip: rect(0, 0, 0, 0); border: 0; }
        .t-12 { font-size: 12px; }
        .block { display: block; }
        .t-black--light { color: #555; }
        .t-normal { font-weight: normal; }
        .pseudo-generated-class-xyz { /* Unstable */ }
    </style>
</head>
<body>
    <header id="main-header" class="site-header pseudo-generated-class-xyz">
        <h1>Complex Page - Stability Test</h1>
        <nav aria-label="Primary Navigation">
            <a class="CBbYpNePzBFhWBvcNlkynoaEFSgUIe global-nav__primary-link global-nav__primary-link--active" href="/home" aria-current="page">
                <span class="t-12 block t-black--light t-normal">Home</span>
            </a>
            <a id="ember12" class="CBbYpNePzBFhWBvcNlkynoaEFSgUIe global-nav__primary-link ember-view" href="/network">
                <span class="t-12 block t-black--light t-normal">My Network</span>
                <span class="notification-badge notification-badge--show">
                    <span aria-hidden="true">2</span>
                    <span class="visually-hidden">2 new network updates</span>
                </span>
            </a>
            <a class="CBbYpNePzBFhWBvcNlkynoaEFSgUIe global-nav__primary-link" href="/jobs">
                 <span class="t-12 block t-black--light t-normal">Jobs</span>
            </a>
            <a class="CBbYpNePzBFhWBvcNlkynoaEFSgUIe global-nav__primary-link" href="/messaging/">
                 <span class="t-12 block t-black--light t-normal">Messaging</span>
            </a>
            <a id="ember15" class="aBcXyZ123 global-nav__primary-link ember-view" href="/notifications">
                 <span class="t-12 block t-black--light t-normal">Notifications</span>
                 <span class="notification-badge notification-badge--show" aria-hidden="true">1</span>
                 <span class="visually-hidden">1 new notification</span>
            </a>
             <button class="search-global-typeahead__collapsed-search-button" aria-label="Click to start a search" type="button">Search</button>
        </nav>
    </header>

    <div class="container" role="main">
        <section id="section-a" class="section-a">
            <h2>Section A</h2>
            <p>Some introductory text for section A.</p>
            <div class="item-list">
                <div id="ember21" class="item pseudo-generated-class-xyz ember-view">Item A1 <span class="data-point">Data A1</span></div>
                <div class="item highlight" data-stable-id="item-a2">Item A2 (Highlighted) <span class="data-point">Data A2</span></div>
                <div class="item">Item A3 <span class="data-point">Data A3</span></div>
            </div>
        </section>

        <section id="section-b" class="section-b">
            <h2>Section B</h2>
             <div class="item-list">
                 <div class="item" role="listitem">Item B1 <span class="data-point">Data B1</span></div>
                 <div id="ember28" class="item pseudo-generated-class-xyz">Item B2 <span class="data-point">Data B2</span></div>
                 <a href="#target" class="item btn btn-primary CBbYpNePzBFhWBvcNlkynoaEFSgUIe">Link B3</a>
             </div>
        </section>
    </div>

    <footer>
        <p>&copy; 2024 Stability Test Inc.</p>
    </footer>
</body>
</html>
"""


# --- Main Execution Logic ---


async def main():
    # Define target description and extraction goal
    target_description = "the 'Notifications' link in the primary navigation"
    extraction_attribute = "href"
    # Optional: Text to help verify the element if description is ambiguous
    verification_text = "Notifications"

    # Check for API key (Agent constructor handles this now, but keep check for early exit)
    if not os.getenv("OPENAI_API_KEY"):
        # Use rich print if available, otherwise standard print
        try:
            from rich import print

            print("\n[bold red]Error:[/bold red] OPENAI_API_KEY environment variable not set.")
        except ImportError:
            print("\nError: OPENAI_API_KEY environment variable not set.")
        return

    # Use rich print if available
    try:
        from rich import print
    except ImportError:
        pass  # Fallback to standard print

    print(
        f"\n Agent Goal: Find a stable selector for '{target_description}', extract its '{extraction_attribute}' attribute."
    )
    print("\n🚀 Instantiating and running agent...")

    try:
        # Instantiate the agent with the HTML content
        # Note: API key is handled by the constructor via env var or explicit pass
        agent_instance = SelectorAgent(html_content=DUMMY_HTML)

        # Run the agent's find_and_extract method
        result: AgentResult = await agent_instance.find_and_extract(
            target_description=target_description,
            attribute_to_extract=extraction_attribute,
            verification_text=verification_text,
            # extract_text can be omitted, defaults handled in method
        )

        print("\n✅ Agent finished.")
        print("\n--- Final Agent Result ---")

        # Process and print the AgentResult
        print(f"Proposed Selector: {result.proposed_selector}")
        print(f"Reasoning: {result.reasoning}")
        print("--- Extraction --- ")
        print(f"  Attribute Requested: {result.attribute_extracted}")
        print(f"  Text Requested: {result.text_extracted_flag}")
        if result.extraction_result.error:
            print(f"  Extraction Error: {result.extraction_result.error}")
        else:
            if result.attribute_extracted:
                print(
                    f"  Extracted '{result.attribute_extracted}' Value: {result.extraction_result.extracted_attribute_value}"
                )
            if result.text_extracted_flag:
                print(f"  Extracted Text: {result.extraction_result.extracted_text}")

        print("--- Final Verification --- ")
        eval_result = result.final_verification
        print(f"  Selector Used: {eval_result.selector_used}")
        print(f"  Anchor Used: {eval_result.anchor_selector_used}")
        print(f"  Element Count: {eval_result.element_count}")
        first_match = eval_result.matches[0] if eval_result.matches else None
        if first_match:
            print(f"  First Match Tag: '{first_match.tag_name}'")
            print(f"  First Match Text: '{first_match.text_content}'")
            print(f"  First Match Attrs: {first_match.attributes}")
        else:
            print("  First Match: None")
        print(f"  Target Text Found Flag: {eval_result.target_text_found_in_any_match}")
        if eval_result.error:
            print(f"  Evaluation Error: {eval_result.error}")

        # Determine overall success based on extraction and verification
        extraction_successful = not result.extraction_result.error and (
            (
                result.attribute_extracted
                and result.extraction_result.extracted_attribute_value is not None
            )
            or (result.text_extracted_flag and result.extraction_result.extracted_text is not None)
        )
        # Adjust verification success slightly - focus on count=1 and no error, text match is secondary if selector is good
        verification_successful = eval_result.element_count == 1 and not eval_result.error

        if extraction_successful and verification_successful:
            print(
                "[bold green]Overall SUCCESS: Unique element found and data extracted correctly.[/bold green]"
            )
        elif verification_successful:
            print(
                "[bold yellow]Overall PARTIAL: Unique element found, but data extraction failed or yielded no value.[/bold yellow]"
            )
        else:
            print(
                "[bold red]Overall FAILED: Element not found uniquely, verification failed, or agent error.[/bold red]"
            )

    except ValueError as _ve:
        print(f"\n[bold red]Initialization Error:[/bold red] {_ve}")
    except Exception as _e:
        print(f"\n[bold red]Error during agent execution:[/bold red] {type(_e).__name__}: {_e}")


if __name__ == "__main__":
    asyncio.run(main())
