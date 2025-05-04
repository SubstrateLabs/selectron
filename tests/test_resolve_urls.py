import pytest
from bs4 import BeautifulSoup

from selectron.util.resolve_urls import resolve_urls


@pytest.mark.parametrize(
    "html_input, base_url, expected_output",
    [
        # Basic case: relative link and image
        (
            '<div><a href="/about">About</a><img src="logo.png"></div>',
            "http://example.com",
            '<div><a href="http://example.com/about">About</a><img src="http://example.com/logo.png"/></div>',
        ),
        # Absolute link and image (should remain unchanged)
        (
            '<div><a href="http://othersite.com/page">Other</a><img src="https://secure.com/img.jpg"></div>',
            "http://example.com",
            '<div><a href="http://othersite.com/page">Other</a><img src="https://secure.com/img.jpg"/></div>',
        ),
        # Mixed content
        (
            '<div><a href="/terms">Terms</a><img src="https://cdn.example.net/icon.svg"></div>',
            "http://example.com/nested/",
            '<div><a href="http://example.com/terms">Terms</a><img src="https://cdn.example.net/icon.svg"/></div>',
        ),
        # Link with query parameters
        (
            '<div><a href="search?q=test">Search</a></div>',
            "http://example.com/app/",
            '<div><a href="http://example.com/app/search?q=test">Search</a></div>',
        ),
        # Empty href/src
        (
            '<div><a href="">Empty Link</a><img src=""/></div>',
            "http://example.com",
            '<div><a href="">Empty Link</a><img src=""/></div>',
        ),
        # No relevant tags
        (
            "<div><p>Just text</p><span>Span</span></div>",
            "http://example.com",
            "<div><p>Just text</p><span>Span</span></div>",
        ),
        # Malformed url (should log warning but proceed)
        (
            '<a href="/path with space">Link</a>',
            "http://example.com",
            '<a href="http://example.com/path with space">Link</a>',
        ),
    ],
)
def test_resolve_urls(html_input, base_url, expected_output):
    resolved_html = resolve_urls(html_input, base_url)
    # Use BeautifulSoup to parse both to normalize minor HTML differences
    resolved_soup = BeautifulSoup(resolved_html, "html.parser")
    expected_soup = BeautifulSoup(expected_output, "html.parser")
    assert str(resolved_soup) == str(expected_soup)


def test_resolve_urls_invalid_base():
    """Test with an invalid base URL - should still process but might log errors."""
    html_input = '<a href="/relative">Rel</a>'
    base_url = "not_a_url"
    # urljoin doesn't join with invalid base, so href remains relative
    expected_output = '<a href="/relative">Rel</a>'
    resolved_html = resolve_urls(html_input, base_url)
    resolved_soup = BeautifulSoup(resolved_html, "html.parser")
    expected_soup = BeautifulSoup(expected_output, "html.parser")
    assert str(resolved_soup) == str(expected_soup)
