import json  # Need this
from importlib.abc import Traversable  # Need this for mock spec
from unittest.mock import MagicMock

import pytest

from selectron.parse.parser_fallback import find_fallback_parser
from selectron.util.slugify_url import slugify_url  # Need this to create slug keys


@pytest.mark.parametrize(
    "target_url, available_parsers_content, expected_data",  # Renamed/removed params
    [
        # 1. Exact match
        (
            "https://example.com/path/to/page",
            # Map slugs to content dicts
            {slugify_url("https://example.com/path/to/page"): {"data": "exact"}},
            {"data": "exact"},
        ),
        # 2. Fallback to parent
        (
            "https://example.com/path/to/page",
            {slugify_url("https://example.com/path/to"): {"data": "parent"}},
            {"data": "parent"},
        ),
        # 3. Fallback to grandparent
        (
            "https://example.com/path/to/page",
            {slugify_url("https://example.com/path"): {"data": "grandparent"}},
            {"data": "grandparent"},
        ),
        # 4. Fallback to domain root (with trailing slash slug)
        (
            "https://example.com/path/to/page",
            {slugify_url("https://example.com/"): {"data": "root_slash"}},
            {"data": "root_slash"},
        ),
        # 5. Fallback to domain root (without trailing slash slug)
        (
            "https://example.com/path",
            {slugify_url("https://example.com"): {"data": "root_no_slash"}},
            {"data": "root_no_slash"},
        ),
        # 6. No match found
        (
            "https://example.com/path/to/page",
            {slugify_url("https://another.com/"): {"data": "other"}},
            None,
        ),
        # 7. URL with no path, exact match
        (
            "https://example.com",
            {slugify_url("https://example.com"): {"data": "no_path_exact"}},
            {"data": "no_path_exact"},
        ),
        # 8. URL with no path, no match
        ("https://example.com", {}, None),
        # 9. URL with trailing slash, match parent (domain root without slash)
        (
            "https://example.com/path/",
            {slugify_url("https://example.com"): {"data": "root_match"}},
            {"data": "root_match"},
        ),
        # 10. URL with query/fragment (should be ignored by fallback, match base)
        (
            "http://test.com/a/b?query=1#frag",
            {slugify_url("http://test.com/a"): {"data": "base_a"}},
            {"data": "base_a"},
        ),
        # 11. Deep path fallback
        (
            "https://host.com/a/b/c/d/e",
            {slugify_url("https://host.com/a/b"): {"data": "deep"}},
            {"data": "deep"},
        ),
        # 12. URL like file path fallback
        (
            "file:///path/to/resource",
            {slugify_url("file:///path/to"): {"data": "file_parent"}},
            {"data": "file_parent"},
        ),
        # 13. Domain fallback: no standard match, finds parser on same domain/different path
        (
            "https://example.com/user/profile",
            {
                slugify_url("https://another.com/"): {"data": "other"},
                slugify_url("https://example.com/settings"): {"data": "domain_sibling"},
            },
            {"data": "domain_sibling"},
        ),
        # 14. Domain fallback: prioritizes shallower path
        (
            "https://example.com/user/profile/deep",
            {
                slugify_url("https://example.com/settings/advanced"): {"data": "domain_deep"},
                slugify_url("https://example.com/settings"): {"data": "domain_shallow"},
            },
            {"data": "domain_shallow"},  # Should pick settings over settings/advanced
        ),
        # 15. Domain fallback: standard path fallback wins over domain fallback
        (
            "https://example.com/user/profile",
            {
                slugify_url("https://example.com/user"): {"data": "path_parent"},
                slugify_url("https://example.com/settings"): {"data": "domain_sibling"},
            },
            {"data": "path_parent"},  # Path fallback should be used first
        ),
        # 16. Domain fallback: no matching domain
        (
            "https://example.com/user/profile",
            {slugify_url("https://another.com/settings"): {"data": "other_domain"}},
            None,
        ),
        # 17. Domain fallback: file URI finds sibling
        (
            "file:///user/data/config.txt",
            {slugify_url("file:///user/logs/log.txt"): {"data": "file_sibling"}},
            {"data": "file_sibling"},
        ),
        # 18. Domain fallback: file URI prioritizes shallower
        (
            "file:///user/data/deep/config.txt",
            {
                slugify_url("file:///user/logs/archive/old.log"): {"data": "file_deep"},
                slugify_url("file:///user/logs"): {"data": "file_shallow"},
            },
            {"data": "file_shallow"},
        ),
    ],
)
def test_find_fallback_parser(
    target_url: str,
    available_parsers_content: dict[str, dict],  # Slug -> Content Dict
    expected_data: dict | None,
):
    """Tests the find_fallback_parser utility with various scenarios."""
    # Create mock Traversable objects for the available parser content
    available_parsers_with_mocks: dict[str, MagicMock] = {}
    for slug, content_dict in available_parsers_content.items():
        mock_ref = MagicMock(spec=Traversable)
        # Configure read_text to return the JSON string of the content
        mock_ref.read_text.return_value = json.dumps(content_dict)
        available_parsers_with_mocks[slug] = mock_ref

    # Call the function with the dictionary of mock Traversables
    result = find_fallback_parser(target_url, available_parsers_with_mocks)  # type: ignore

    # Assert the final loaded data is correct
    assert result == expected_data

    # Removed assertions checking internal mock calls, as loading is now internal
