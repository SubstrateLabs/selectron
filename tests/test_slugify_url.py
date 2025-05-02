import re

import pytest

from selectron.util.slugify_url import slugify_url, unslugify_url


@pytest.mark.parametrize(
    "original_url, expected_slug",
    [
        # Basic cleaning and encoding
        ("https://www.example.com/path?query=1#hash", "example~~2ecom~~2fpath"),
        (
            "http://example.com/another/path/",
            "example~~2ecom~~2fanother~~2fpath",
        ),  # trailing slash removed
        ("example.com/index.html", "example~~2ecom"),
        ("example.com//double//slash", "example~~2ecom~~2fdouble~~2fslash"),  # slash normalization
        # Basic encoding
        ("user:pass@example.com", "user~~3apass~~40example~~2ecom"),
        ("example.com/file.name", "example~~2ecom~~2ffile~~2ename"),
        ("example.com/path with space", "example~~2ecom~~2fpath~~20with~~20space"),
        # Other printable ASCII (e.g., underscore, plus)
        ("example.com/path_plus+test", "example~~2ecom~~2fpath~~5fplus~~2btest"),
        # Idempotency (partially slugged - simple version doesn't check this upfront)
        (
            "example~~2ecom/path",
            "example~~7e~~7e2ecom~~2fpath",
        ),  # Note: Simple version re-encodes the ~~
    ],
)
def test_slugify_url_simple(original_url, expected_slug):
    assert slugify_url(original_url) == expected_slug


# Test the round trip: simple slugify -> simple unslugify
@pytest.mark.parametrize(
    "original_url",
    [
        "https://www.example.com/path?query=1#hash",
        "http://example.com/another/path/",
        "example.com/file.name",
        "user:pass@example.com/path with space",
        "example.com/path_plus+test",
    ],
)
def test_simple_slugify_unslugify_roundtrip(original_url):
    # 1. Calculate the expected result after simple cleaning (what slugify *should* preserve)
    expected = original_url
    expected = re.sub(r"^https?:\/\/", "", expected)
    expected = re.sub(r"^www\.", "", expected)
    expected = expected.split("?")[0].split("#")[0]
    expected = re.sub(r"/+$", "", expected)
    expected = re.sub(r"/index\.html$", "", expected)
    expected = re.sub(r"/+", "/", expected)
    # We don't add back protocol in the simple version

    # 2. Perform the round trip
    slug = slugify_url(original_url)
    unslugged = unslugify_url(slug)

    # 3. Compare
    # Need to handle the case where original contained chars that get encoded, like space or underscore
    # The 'expected' variable has the cleaned *original* chars, 'unslugged' has the decoded chars.
    assert unslugged == expected
