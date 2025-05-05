import json
from pathlib import Path

from selectron.lib import parse

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_parse_x_home():
    """Tests parsing a single x.com/home tweet element using the library function."""
    fixture_path = FIXTURES_DIR / "x.com~~2fhome.json"
    url = "https://x.com/home"  # The URL dictates which parser is used

    with open(fixture_path, "r") as f:
        fixture_data = json.load(f)

    # Use the first HTML element from the fixture
    html_content = fixture_data["html_elements"][0]

    results = parse(url=url, html_content=html_content)

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

    # TODO: parse metrics
    # assert isinstance(tweet_data["reply_count"], int)
    # if "repost_count" in tweet_data:
    # assert isinstance(tweet_data["repost_count"], int)
    # if "like_count" in tweet_data:
    # assert isinstance(tweet_data["like_count"], int)
    # if "view_count" in tweet_data:
    # assert isinstance(tweet_data["view_count"], int)
