import os

import pytest

from selectron.ai.selector_agent import AgentResult, SelectorAgent

# HTML Snippet mimicking a Twitter-like post structure
TWITTER_HTML_SNIPPET = """
<article aria-labelledby="id-z241s5-author id-z241s5-content" role="article" tabindex="-1"
    class="css-175oi2r r-14gqq1x r-1q9bdsx r-ltgprq r-1udh08x r-u8s1d r-1habvwh r-3s2u2q r-6qh972 r-1loqt21 r-1ny4l3l r-18u37iz r-oyd9sg r-13qz1uu r-qklmqi r-1wbh5a2 r-1sp51qo r-mk0yit r-1j63xyz"
    data-testid="tweet">
    <div class="css-175oi2r r-1d09ksm r-18u37iz r-1wbh5a2">
        <!-- User Info Section -->
        <div class="css-175oi2r r-16y2uox r-1wbh5a2 r-z32n2g r-sdzlij r-1udh08x r-aqfbo4 r-u8s1d r-ipm5af r-1jjv7ws r-1adg3ll r-1ny4l3l"
            data-testid="User-Name">
            <div class="css-175oi2r r-1wbh5a2 r-dnmrzs r-1ny4l3l r-1udh08x r-1sp51qo">
                <div class="css-175oi2r r-1awozwy r-z32n2g r-18u37iz r-6qh972">
                    <div class="css-175oi2r r-1wbh5a2 r-dnmrzs r-1ny4l3l r-1loqt21">
                        <a href="/SomeUser" role="link" tabindex="-1" class="css-175oi2r r-1loqt21 r-1wbh5a2 r-dnmrzs r-1ny4l3l">
                            <div class="css-175oi2r r-1awozwy r-z32n2g r-18u37iz">
                                <div dir="ltr" class="css-1rynq56 r-dnmrzs r-1udh08x r-1sp51qo r-1j63xyz r-bcqeeo r-qvutc0">
                                    <span class="css-1qaijid r-bcqeeo r-qvutc0 r-poiln3">
                                        <span class="css-1qaijid r-bcqeeo r-qvutc0 r-poiln3">Example Name</span>
                                    </span>
                                </div>
                                <div class="css-175oi2r r-1wbh5a2 r-dnmrzs r-1ny4l3l">
                                    <div class="css-175oi2r r-18u37iz r-dnmrzs">
                                        <!-- Potentially contains verification icon etc. -->
                                    </div>
                                </div>
                            </div>
                        </a>
                    </div>
                </div>
                <div class="css-175oi2r r-1awozwy r-z32n2g r-18u37iz">
                    <div class="css-175oi2r r-1wbh5a2 r-dnmrzs r-1ny4l3l r-1loqt21">
                        <a href="/SomeUser" role="link" tabindex="-1" class="css-175oi2r r-1loqt21 r-1wbh5a2 r-dnmrzs r-1ny4l3l">
                            <div dir="ltr" class="css-1rynq56 r-18u37iz r-1q142lx r-xoduu5 r-1sp51qo r-1j63xyz r-bcqeeo r-qvutc0">
                                <span class="css-1qaijid r-bcqeeo r-qvutc0 r-poiln3">@SomeUser</span>
                            </div>
                        </a>
                    </div>
                    <div class="css-1rynq56 r-1wbh5a2 r-dnmrzs r-1ny4l3l r-1loqt21 r-u8s1d r-1j63xyz r-bcqeeo r-qvutc0">
                        <span class="css-1qaijid r-bcqeeo r-qvutc0 r-poiln3">·</span>
                    </div>
                    <a href="/SomeUser/status/1234567890123456789" class="css-1rynq56 r-1loqt21 r-1wbh5a2 r-dnmrzs r-1ny4l3l r-u8s1d r-1j63xyz r-bcqeeo r-qvutc0"
                        role="link" tabindex="-1">
                        <time datetime="2024-01-01T12:00:00.000Z">Jan 1</time>
                    </a>
                </div>
            </div>
        </div>

        <!-- Tweet Content Section -->
        <div class="css-175oi2r r-18u37iz r-1q142lx">
            <div lang="en" dir="ltr" class="css-1rynq56 r-18u37iz r-1sp51qo r-1j63xyz r-1kqtdi0 r-bcqeeo r-qvutc0"
                id="id-z241s5-content" data-testid="tweetText">
                <span class="css-1qaijid r-bcqeeo r-qvutc0 r-poiln3">This is the main text content of the tweet. It might mention
                    <a href="/AnotherUser" role="link" class="css-1qaijid r-1loqt21 r-1vr29t4 r-1kyxqc6 r-1j63xyz r-13hce6t r-o7ynqc r-f6qlzz r-1dr7s14 r-bcqeeo r-poiln3 r-qvutc0">
                        <span class="css-1qaijid r-bcqeeo r-qvutc0 r-poiln3">@AnotherUser</span></a> or include
                    <a href="/search?q=%23hashtag" role="link" class="css-1qaijid r-1loqt21 r-1vr29t4 r-1kyxqc6 r-1j63xyz r-13hce6t r-o7ynqc r-f6qlzz r-1dr7s14 r-bcqeeo r-poiln3 r-qvutc0">#hashtag</a>.
                    Check out <a href="https://example.com" rel="nofollow noopener noreferrer" target="_blank" role="link" class="css-1qaijid r-1loqt21 r-1vr29t4 r-1kyxqc6 r-1j63xyz r-13hce6t r-o7ynqc r-f6qlzz r-1dr7s14 r-bcqeeo r-poiln3 r-qvutc0">https://example.com</a>.
                </span>
            </div>
        </div>

        <!-- Media Section (Optional) -->
        <div class="css-175oi2r r-1kqtdi0 r-1867qdf r-1sp51qo r-1j63xyz r-18u37iz r-61z16t r-1pi2tsx r-1ny4l3l r-o7ynqc r-1dr7s14 r-1adg3ll" data-testid="tweetPhoto">
            <div class="css-175oi2r r-1p0dtai r-1loqt21 r-1pi2tsx r-u8s1d r-1ny4l3l r-1k1k8z7 r-o7ynqc r-6416eg r-1dr7s14 r-u1s4t0 r-zchlnj r-ipm5af r-13qz1uu r-1wyyakw">
                <a href="/SomeUser/status/1234567890123456789/photo/1" role="link" class="css-175oi2r r-1p0dtai r-1loqt21 r-1adg3ll r-1ny4l3l r-o7ynqc r-6416eg r-1dr7s14 r-1enofrn r-zchlnj r-ipm5af r-u1s4t0 r-13qz1uu r-1wyyakw">
                    <div class="css-175oi2r r-1p0dtai r-1adg3ll r-1pi2tsx r-1wy6sm r-u8s1d r-zchlnj r-ipm5af r-13qz1uu">
                        <img alt="Image" draggable="true"
                             src="https://pbs.twimg.com/media/GExampleMediaId?format=jpg&name=medium"
                             class="css-1qaijid r-1pi2tsx r-1wy6sm r-u8s1d r-1b7u577 r-o7ynqc r-6416eg r-lrvibr r-1ipicw7">
                    </div>
                </a>
            </div>
        </div>

        <!-- Action Bar -->
        <div class="css-175oi2r r-1kbdv8c r-18u37iz r-1wtj0ep r-1s2bzr4 r-1ye8kvj r-zl2h9q">
            <div role="group" class="css-175oi2r r-1kbdv8c r-18u37iz r-1wtj0ep r-1s2bzr4 r-1ye8kvj">
                <button data-testid="reply" class="css-175oi2r r-xoduu5 r-42olwf r-sdzlij r-1phboty r-rs99b7 r-1sp51qo r-2yi16 r-1qi8awa r-1ny4l3l r-ymttw5 r-o7ynqc r-6416eg r-lrvibr">
                    <!-- Reply icon and count -->
                    <span data-testid="app-text-transition-container" style="transition-property: transform; transition-duration: 0.3s;">
                        <span class="css-1qaijid r-1b43r93 r-bcqeeo r-qvutc0 r-poiln3">
                            <span class="css-1qaijid r-bcqeeo r-qvutc0 r-poiln3">123</span>
                        </span>
                    </span>
                </button>
                <button data-testid="retweet" class="css-175oi2r r-xoduu5 r-42olwf r-sdzlij r-1phboty r-rs99b7 r-1sp51qo r-2yi16 r-1qi8awa r-1ny4l3l r-ymttw5 r-o7ynqc r-6416eg r-lrvibr">
                    <!-- Retweet icon and count -->
                     <span data-testid="app-text-transition-container" style="transition-property: transform; transition-duration: 0.3s;">
                        <span class="css-1qaijid r-1b43r93 r-bcqeeo r-qvutc0 r-poiln3">
                            <span class="css-1qaijid r-bcqeeo r-qvutc0 r-poiln3">1.5K</span>
                        </span>
                    </span>
                </button>
                <button data-testid="like" class="css-175oi2r r-xoduu5 r-42olwf r-sdzlij r-1phboty r-rs99b7 r-1sp51qo r-2yi16 r-1qi8awa r-1ny4l3l r-ymttw5 r-o7ynqc r-6416eg r-lrvibr">
                    <!-- Like icon and count -->
                     <span data-testid="app-text-transition-container" style="transition-property: transform; transition-duration: 0.3s;">
                        <span class="css-1qaijid r-1b43r93 r-bcqeeo r-qvutc0 r-poiln3">
                            <span class="css-1qaijid r-bcqeeo r-qvutc0 r-poiln3">10M</span>
                        </span>
                    </span>
                </button>
                 <a href="/SomeUser/status/1234567890123456789/analytics" class="css-175oi2r r-xoduu5 r-42olwf r-sdzlij r-1phboty r-rs99b7 r-1sp51qo r-2yi16 r-1qi8awa r-1ny4l3l r-ymttw5 r-o7ynqc r-6416eg r-lrvibr">
                    <!-- Views/Analytics icon and count -->
                    <span class="css-1qaijid r-1b43r93 r-bcqeeo r-qvutc0 r-poiln3">
                        <span class="css-1qaijid r-bcqeeo r-qvutc0 r-poiln3">50.2K</span> Views
                    </span>
                </a>
            </div>
        </div>
    </div>
</article>
"""

openai_api_key_present = bool(os.getenv("OPENAI_API_KEY"))
skip_if_no_key = pytest.mark.skipif(
    not openai_api_key_present,
    reason="Requires OPENAI_API_KEY environment variable",
)


# --- Test Case 1: Select Tweet Container ---
@skip_if_no_key
@pytest.mark.asyncio
async def test_agent_selects_tweet_container():
    """Tests if the agent can select the main tweet container using stable attributes."""
    # --- Arrange ---
    target_description = "the main container for the tweet"
    # No extraction needed, just selection
    extraction_attribute = None
    extract_text = False
    # Verification text could target something reliably inside the article
    verification_text = "@SomeUser"
    base_url = "https://x.com"  # Provide base_url

    agent_instance = SelectorAgent(html_content=TWITTER_HTML_SNIPPET, base_url=base_url)

    # --- Act ---
    result: AgentResult = await agent_instance.find_and_extract(
        target_description=target_description,
        attribute_to_extract=extraction_attribute,
        extract_text=extract_text,
        verification_text=verification_text,
    )

    # --- Assert ---
    assert result.final_verification.element_count == 1, "Verification failed: Element not unique"
    assert result.final_verification.error is None, (
        f"Verification failed: {result.final_verification.error}"
    )
    # Check the selector targets the article or uses the data-testid
    assert (
        "article" in result.proposed_selector or '[data-testid="tweet"]' in result.proposed_selector
    ), "Selector does not seem to target the article or use data-testid='tweet'"
    # Check if the correct element was found (basic tag check)
    assert result.final_verification.matches, "Verification failed: No match details found"
    assert result.final_verification.matches[0].tag_name == "article", (
        "Verification failed: Selected element is not an article"
    )

    # Ensure unstable classes are avoided (example from HTML)
    assert "css-175oi2r" not in result.proposed_selector, "Selector uses unstable CSS class"
    assert result.attribute_extracted is None
    assert result.text_extracted_flag is False


# --- Test Case 2: Extract Tweet URL ---
@skip_if_no_key
@pytest.mark.asyncio
async def test_agent_extracts_tweet_url():
    """Tests if the agent can find the tweet's canonical URL (timestamp link)."""
    # --- Arrange ---
    target_description = "the timestamp link that contains the tweet URL"
    extraction_attribute = "href"
    extract_text = False
    # Use time as verification hint
    verification_text = "Jan 1"
    base_url = "https://x.com"
    expected_url_path = "/SomeUser/status/1234567890123456789"
    expected_absolute_url = f"{base_url}{expected_url_path}"  # Expect absolute URL

    agent_instance = SelectorAgent(html_content=TWITTER_HTML_SNIPPET, base_url=base_url)

    # --- Act ---
    result: AgentResult = await agent_instance.find_and_extract(
        target_description=target_description,
        attribute_to_extract=extraction_attribute,
        extract_text=extract_text,
        verification_text=verification_text,
    )

    # --- Assert ---
    assert result.final_verification.element_count == 1, "Verification failed: Element not unique"
    assert result.final_verification.error is None, (
        f"Verification failed: {result.final_verification.error}"
    )
    assert result.extraction_result.error is None, (
        f"Extraction failed: {result.extraction_result.error}"
    )
    # Assert against the expected ABSOLUTE url
    assert result.extraction_result.extracted_attribute_value == expected_absolute_url, (
        f"Extraction failed: Incorrect absolute href extracted. Got: {result.extraction_result.extracted_attribute_value}"
    )

    # Check selector stability / target (href check might need adjustment)
    assert (
        "time" in result.proposed_selector
        or "datetime" in result.proposed_selector
        # Check for relative or absolute path in selector
        or f"[href='{expected_url_path}']" in result.proposed_selector
        or f'[href="{expected_url_path}"]' in result.proposed_selector
        or f"[href='{expected_absolute_url}']" in result.proposed_selector
        or f'[href="{expected_absolute_url}"]' in result.proposed_selector
        or "/status/" in result.proposed_selector
    ), "Selector does not seem related to the timestamp or status link"
    assert '[data-testid="User-Name"]' not in result.proposed_selector, (
        "Selector might be targeting the whole user section, not the specific timestamp link"
    )
    assert "css-1rynq56" not in result.proposed_selector, "Selector uses unstable CSS class"

    assert result.attribute_extracted == extraction_attribute
    assert result.text_extracted_flag is False


# --- Test Case 3: Extract Tweet Text ---
@skip_if_no_key
@pytest.mark.asyncio
async def test_agent_extracts_tweet_text():
    """Tests if the agent can find and extract the main tweet text content."""
    # --- Arrange ---
    target_description = "the main text content of the tweet"
    extraction_attribute = None
    extract_text = True
    # Use a snippet of the text for verification
    verification_text = "This is the main text content"
    # Adjust expected text to match get_text(separator=' ') behavior
    expected_text = "This is the main text content of the tweet. It might mention @AnotherUser or include #hashtag . Check out https://example.com ."
    # Expect markdown with absolutified links AND correct external link format
    expected_markdown = "This is the main text content of the tweet. It might mention [@AnotherUser](https://x.com/AnotherUser) or include [#hashtag](https://x.com/search?q=%23hashtag). Check out <https://example.com>."
    base_url = "https://x.com"

    agent_instance = SelectorAgent(html_content=TWITTER_HTML_SNIPPET, base_url=base_url)

    # --- Act ---
    result: AgentResult = await agent_instance.find_and_extract(
        target_description=target_description,
        attribute_to_extract=extraction_attribute,
        extract_text=extract_text,
        verification_text=verification_text,
    )

    # --- Assert ---
    print("\n--- Extracted Markdown Content ---")
    print(result.extraction_result.markdown_content)
    print("------------------------------------\n")

    assert result.final_verification.element_count == 1, "Verification failed: Element not unique"
    assert result.final_verification.error is None, (
        f"Verification failed: {result.final_verification.error}"
    )
    assert result.extraction_result.error is None, (
        f"Extraction failed: {result.extraction_result.error}"
    )
    # Check extracted text is not None before trying to split
    assert result.extraction_result.extracted_text is not None, (
        "Extraction failed: extracted_text is None"
    )
    # Normalize whitespace for plain text comparison
    extracted_normalized = " ".join(result.extraction_result.extracted_text.split())
    expected_normalized = " ".join(expected_text.split())
    assert extracted_normalized == expected_normalized, (
        f"Extraction failed: Incorrect plain text extracted.\nExpected: {expected_normalized}\nGot: {extracted_normalized}"
    )

    # Check markdown content
    assert result.extraction_result.markdown_content is not None, (
        "Extraction failed: markdown_content is None"
    )
    # Normalize whitespace for markdown comparison too
    extracted_md_normalized = " ".join(result.extraction_result.markdown_content.split())
    expected_md_normalized = " ".join(expected_markdown.split())
    assert extracted_md_normalized == expected_md_normalized, (
        f"Extraction failed: Incorrect markdown content.\nExpected: {expected_md_normalized}\nGot: {extracted_md_normalized}"
    )

    # Check selector stability / target
    assert (
        '[data-testid="tweetText"]' in result.proposed_selector
        or "[data-testid='tweetText']" in result.proposed_selector
        or 'id="id-z241s5-content"'
        in result.proposed_selector  # Less ideal but potentially stable ID
    ), "Selector does not seem to target data-testid='tweetText' or its ID"
    assert "css-1rynq56" not in result.proposed_selector, "Selector uses unstable CSS class"

    assert result.attribute_extracted is None
    assert result.text_extracted_flag is True
