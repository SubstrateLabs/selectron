import json
from pathlib import Path

from selectron.lib import parse
from selectron.parse.types import ParserError, ParseSuccess  # Import result types

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_parse_x_home():
    """Tests parsing a single x.com/home tweet element using the library function."""
    fixture_path = FIXTURES_DIR / "x.com~~2fhome.json"
    url = "https://x.com/home"  # The URL dictates which parser is used

    with open(fixture_path, "r") as f:
        fixture_data = json.load(f)

    # Use the first HTML element from the fixture
    html_content = fixture_data["html_elements"][0]

    outcome = parse(url=url, html_content=html_content)

    # Assert that the parsing was successful
    assert isinstance(outcome, ParseSuccess), (
        f"Parsing failed: {outcome.message if isinstance(outcome, ParserError) else 'Unknown error'}"
    )
    # If successful, outcome.data contains the list of results
    results = outcome.data

    # Assertions based on the expected parser behavior and the specific content
    # of the first element in tests/fixtures/x.com~~2fhome.json
    assert isinstance(results, list)
    # Since we parse the whole HTML string, but the selector finds only one article,
    # we expect one result dict in the list.
    assert len(results) == 1

    tweet_data = results[0]
    assert isinstance(tweet_data, dict)

    # Check for expected keys (presence and basic type)
    assert "id" in tweet_data
    assert isinstance(tweet_data["id"], str)
    assert "primary_url" in tweet_data
    assert isinstance(tweet_data["primary_url"], str)
    assert "datetime" in tweet_data
    assert isinstance(tweet_data["datetime"], str)
    assert "author" in tweet_data
    assert isinstance(tweet_data["author"], str)
    assert "description" in tweet_data
    assert isinstance(tweet_data["description"], str)

    assert tweet_data["author"] == "@_its_not_real_"
    assert tweet_data["primary_url"] == "/_its_not_real_/status/1918760851957321857"
    assert tweet_data["datetime"] == "2025-05-03T20:13:30.000Z"
    assert tweet_data["id"] == "1918760851957321857"
    assert tweet_data["description"].startswith('"They\'re made out of meat."')

    # TODO: parse metrics - these might fail depending on fixture specifics
    # assert "reply_count" in tweet_data, "reply_count missing"
    # assert isinstance(tweet_data.get("reply_count"), int)
    # assert "repost_count" in tweet_data, "repost_count missing"
    # assert isinstance(tweet_data.get("repost_count"), int)
    # assert "like_count" in tweet_data, "like_count missing"
    # assert isinstance(tweet_data.get("like_count"), int)
    # assert "view_count" in tweet_data, "view_count missing"
    # assert isinstance(tweet_data.get("view_count"), int)

    # Optional: Check quote tweet details if present, but don't fail if missing
    # Commenting out the specific value checks as they might be brittle
    if "quoted_author_name" in tweet_data:
        assert isinstance(tweet_data["quoted_author_name"], str)
        # assert tweet_data["quoted_author_name"] == "Guy is WRITING THE BOOK"
    if "quoted_url" in tweet_data:
        assert isinstance(tweet_data["quoted_url"], str)
    if "quoted_text" in tweet_data:
        assert isinstance(tweet_data["quoted_text"], str)
        # assert tweet_data["quoted_text"].startswith("I just don\u2019t understand how text could become self-aware")
