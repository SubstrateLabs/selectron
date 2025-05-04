import json
from importlib.abc import Traversable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from selectron.parse.parser_fallback import find_fallback_parser
from selectron.parse.types import ParserInfo, ParserOrigin
from selectron.util.slugify_url import slugify_url


def create_mock_parser_info(
    slug: str, content_dict: dict | None, origin: ParserOrigin = "source", fail_load: bool = False
) -> ParserInfo:
    """Creates mock ParserInfo. If content_dict is None or fail_load is True, mock loading failure."""
    mock_resource = MagicMock(spec=Traversable)
    if content_dict is None or fail_load:
        # Simulate different load failures
        if fail_load and content_dict is not None:  # Specific request to fail load
            mock_resource.read_text.side_effect = Exception(f"Simulated load failure for {slug}")
        elif isinstance(content_dict, dict):  # Simulate JSON decode error
            mock_resource.read_text.return_value = "invalid json {"
        else:  # Simulate file not found or other read error
            mock_resource.read_text.side_effect = FileNotFoundError(
                f"Simulated not found for {slug}"
            )
    else:
        mock_resource.read_text.return_value = json.dumps(content_dict)

    # Use slug itself in dummy path for easier debugging
    # ParserInfo is type aliased to a tuple: (origin, resource, file_path)
    return (origin, mock_resource, Path(f"dummy/{slug}.json"))


@pytest.mark.parametrize(
    # Keep parameters, add expected_origin, expected_path_part
    "target_url, available_parsers_setup, expected_slug_key, expected_data, expected_origin, expected_path_part",
    [
        # available_parsers_setup: List of tuples (slug, content_dict | None, fail_load) or (slug, content, fail_load, origin)
        # expected_path_part: The slug used in the dummy path, to verify correct ParserInfo returned
        # === Existing Tests (adapted format) ===
        # 1. Exact match
        (
            "https://example.com/path/to/page",
            [(slugify_url("https://example.com/path/to/page"), {"data": "exact"}, False)],
            slugify_url("https://example.com/path/to/page"),
            {"data": "exact"},
            "source",
            slugify_url("https://example.com/path/to/page"),
        ),
        # 2. Fallback to parent
        (
            "https://example.com/path/to/page",
            [(slugify_url("https://example.com/path/to"), {"data": "parent"}, False)],
            slugify_url("https://example.com/path/to"),
            {"data": "parent"},
            "source",
            slugify_url("https://example.com/path/to"),
        ),
        # 3. Fallback to grandparent
        (
            "https://example.com/path/to/page",
            [(slugify_url("https://example.com/path"), {"data": "grandparent"}, False)],
            slugify_url("https://example.com/path"),
            {"data": "grandparent"},
            "source",
            slugify_url("https://example.com/path"),
        ),
        # 4. Fallback to domain root (from path)
        (
            "https://example.com/path/to/page",
            [(slugify_url("https://example.com"), {"data": "root"}, False)],
            slugify_url("https://example.com"),
            {"data": "root"},
            "source",
            slugify_url("https://example.com"),
        ),
        # 5. Fallback to domain root (domain slug matches available domain slug)
        (
            "https://example.com/path",
            [(slugify_url("https://example.com"), {"data": "root_no_slash"}, False)],
            slugify_url("https://example.com"),
            {"data": "root_no_slash"},
            "source",
            slugify_url("https://example.com"),
        ),
        # 6. No match found
        (
            "https://example.com/path/to/page",
            [(slugify_url("https://another.com/"), {"data": "other"}, False)],
            None,
            None,
            None,
            None,
        ),
        # 7. URL with no path, exact match
        (
            "https://example.com",
            [(slugify_url("https://example.com"), {"data": "no_path_exact"}, False)],
            slugify_url("https://example.com"),
            {"data": "no_path_exact"},
            "source",
            slugify_url("https://example.com"),
        ),
        # 8. URL with no path, no match
        ("https://example.com", [], None, None, None, None),
        # 9. URL with trailing slash, match parent (domain root)
        (
            "https://example.com/path/",
            [(slugify_url("https://example.com"), {"data": "root_match"}, False)],
            slugify_url("https://example.com"),
            {"data": "root_match"},
            "source",
            slugify_url("https://example.com"),
        ),
        # 10. URL with query/fragment ignored, match base path
        (
            "http://test.com/a/b?query=1#frag",
            [(slugify_url("http://test.com/a"), {"data": "base_a"}, False)],
            slugify_url("http://test.com/a"),
            {"data": "base_a"},
            "source",
            slugify_url("http://test.com/a"),
        ),
        # 11. Deep path fallback
        (
            "https://host.com/a/b/c/d/e",
            [(slugify_url("https://host.com/a/b"), {"data": "deep"}, False)],
            slugify_url("https://host.com/a/b"),
            {"data": "deep"},
            "source",
            slugify_url("https://host.com/a/b"),
        ),
        # 12. file path fallback
        (
            "file:///path/to/resource",
            [(slugify_url("file:///path/to"), {"data": "file_parent"}, False)],
            slugify_url("file:///path/to"),
            {"data": "file_parent"},
            "source",
            slugify_url("file:///path/to"),
        ),
        # === NEW Sibling Tests ===
        # 13. Sibling match: /user/followers finds /user/verified_followers
        (
            "https://example.com/user/followers",
            [
                (
                    slugify_url("https://example.com/user/verified_followers"),
                    {"data": "sibling"},
                    False,
                )
            ],
            slugify_url("https://example.com/user/verified_followers"),
            {"data": "sibling"},
            "source",
            slugify_url("https://example.com/user/verified_followers"),
        ),
        # 14. Parent wins over sibling
        (
            "https://example.com/user/followers",
            [
                (slugify_url("https://example.com/user"), {"data": "parent"}, False),
                (
                    slugify_url("https://example.com/user/verified_followers"),
                    {"data": "sibling"},
                    False,
                ),
            ],
            slugify_url("https://example.com/user"),
            {"data": "parent"},
            "source",
            slugify_url("https://example.com/user"),
        ),
        # 15. Grandparent wins over sibling-of-parent -> Now sibling wins because parent=/user/profile is checked first
        (
            "https://example.com/user/profile/settings",
            [
                (
                    slugify_url("https://example.com/user"),
                    {"data": "grandparent"},
                    False,
                ),  # Grandparent /user
                (
                    slugify_url("https://example.com/user/profile/activity"),
                    {"data": "sibling"},
                    False,
                ),  # Sibling under /user/profile
            ],
            # Loop checks parent=/user/profile (missing). Check siblings under /user/profile. Finds activity.
            slugify_url("https://example.com/user/profile/activity"),
            {"data": "sibling"},
            "source",
            slugify_url("https://example.com/user/profile/activity"),
        ),
        # 15b. Parent wins over sibling-of-grandparent
        (
            "https://example.com/user/profile/settings",  # Target
            [
                (
                    slugify_url("https://example.com/user/profile"),
                    {"data": "parent"},
                    False,
                ),  # Parent /user/profile
                (
                    slugify_url("https://example.com/user/activity"),
                    {"data": "grandparent_sibling"},
                    False,
                ),  # Sibling under /user
            ],
            # Loop checks parent=/user/profile (found). Returns parent.
            slugify_url("https://example.com/user/profile"),
            {"data": "parent"},
            "source",
            slugify_url("https://example.com/user/profile"),
        ),
        # 16. Sibling found before grandparent (sibling check happens at parent level)
        (
            "https://example.com/catalog/items/123",
            [
                # No parent /catalog/items
                (slugify_url("https://example.com/catalog/items/456"), {"data": "sibling"}, False),
                (slugify_url("https://example.com/catalog"), {"data": "grandparent"}, False),
            ],
            # Loop checks parent=/catalog/items (missing), finds sibling=/catalog/items/456
            slugify_url("https://example.com/catalog/items/456"),
            {"data": "sibling"},
            "source",
            slugify_url("https://example.com/catalog/items/456"),
        ),
        # 17. Multiple siblings, finds first alphabetically by slug
        (
            "https://example.com/data/report",
            [
                (
                    slugify_url("https://example.com/data/summary"),
                    {"data": "sibling_summary"},
                    False,
                ),
                (
                    slugify_url("https://example.com/data/details"),
                    {"data": "sibling_details"},
                    False,
                ),
            ],
            # Parent=/data. Siblings sorted: details, summary. First valid is details.
            slugify_url("https://example.com/data/details"),
            {"data": "sibling_details"},
            "source",
            slugify_url("https://example.com/data/details"),
        ),
        # 18. Root level sibling
        (
            "https://example.com/about",
            [(slugify_url("https://example.com/contact"), {"data": "root_sibling"}, False)],
            # Parent=root. Sibling prefix=https-example-com~~. Finds contact.
            slugify_url("https://example.com/contact"),
            {"data": "root_sibling"},
            "source",
            slugify_url("https://example.com/contact"),
        ),
        # 19. No relevant siblings
        (
            "https://example.com/user/followers",
            [
                (
                    slugify_url("https://example.com/admin/dashboard"),
                    {"data": "irrelevant_sibling"},
                    False,
                ),
                (
                    slugify_url("https://example.com/user-profiles/list"),
                    {"data": "wrong_parent_sibling"},
                    False,
                ),  # Shares domain but not parent prefix /user
            ],
            # Loop checks parent=/user (missing), checks siblings under /user (none). Checks parent=root (missing), checks siblings under root (none).
            None,
            None,
            None,
            None,
        ),
        # 20. file URI sibling
        (
            "file:///data/config/app.yaml",
            [(slugify_url("file:///data/config/user.toml"), {"data": "file_sibling"}, False)],
            slugify_url("file:///data/config/user.toml"),
            {"data": "file_sibling"},
            "source",
            slugify_url("file:///data/config/user.toml"),
        ),
        # 21. file URI parent wins over sibling
        (
            "file:///data/config/app.yaml",
            [
                (slugify_url("file:///data/config"), {"data": "file_parent"}, False),
                (slugify_url("file:///data/config/user.toml"), {"data": "file_sibling"}, False),
            ],
            slugify_url("file:///data/config"),
            {"data": "file_parent"},
            "source",
            slugify_url("file:///data/config"),
        ),
        # 22. Exact match load fails, finds parent (parent check happens before sibling)
        (
            "https://example.com/user/profile",  # Target
            [
                (
                    slugify_url("https://example.com/user/profile"),
                    {"data": "exact"},
                    True,
                ),  # Exact fails load
                (
                    slugify_url("https://example.com/user/settings"),
                    {"data": "sibling"},
                    False,
                ),  # Sibling OK (under parent /user)
                (
                    slugify_url("https://example.com/user"),
                    {"data": "parent"},
                    False,
                ),  # Parent /user OK
            ],
            # Exact match fails. Loop checks parent=/user (found). Returns parent.
            slugify_url("https://example.com/user"),
            {"data": "parent"},
            "source",
            slugify_url("https://example.com/user"),
        ),
        # 22b. Exact match load fails, finds sibling (no parent available)
        (
            "https://example.com/user/profile",  # Target
            [
                (
                    slugify_url("https://example.com/user/profile"),
                    {"data": "exact"},
                    True,
                ),  # Exact fails load
                (
                    slugify_url("https://example.com/user/settings"),
                    {"data": "sibling"},
                    False,
                ),  # Sibling under parent /user
                # No parent /user parser
            ],
            # Exact match fails. Loop checks parent=/user (missing). Checks siblings under /user. Finds settings.
            slugify_url("https://example.com/user/settings"),
            {"data": "sibling"},
            "source",
            slugify_url("https://example.com/user/settings"),
        ),
        # 23. Parent load fails, finds sibling
        (
            "https://example.com/user/profile/view",  # Target
            [
                (
                    slugify_url("https://example.com/user/profile"),
                    {"data": "parent"},
                    True,
                ),  # Parent fails load
                (
                    slugify_url("https://example.com/user/profile/edit"),
                    {"data": "sibling"},
                    False,
                ),  # Sibling OK
            ],
            # Loop checks parent=/user/profile (fails load), checks siblings under /user/profile, finds edit
            slugify_url("https://example.com/user/profile/edit"),
            {"data": "sibling"},
            "source",
            slugify_url("https://example.com/user/profile/edit"),
        ),
        # 24. Sibling load fails, finds next sibling (alphabetical)
        (
            "https://example.com/data/report",  # Target
            [
                (
                    slugify_url("https://example.com/data/details"),
                    {"data": "sibling_details"},
                    True,
                ),  # details fails
                (
                    slugify_url("https://example.com/data/summary"),
                    {"data": "sibling_summary"},
                    False,
                ),  # summary OK
            ],
            # Parent=/data (missing). Checks siblings. details fails load. summary loads ok.
            slugify_url("https://example.com/data/summary"),
            {"data": "sibling_summary"},
            "source",
            slugify_url("https://example.com/data/summary"),
        ),
        # 25. Sibling load fails, finds grandparent
        (
            "https://example.com/catalog/items/123",  # Target
            [
                (
                    slugify_url("https://example.com/catalog/items/456"),
                    {"data": "sibling"},
                    True,
                ),  # Sibling fails
                (
                    slugify_url("https://example.com/catalog"),
                    {"data": "grandparent"},
                    False,
                ),  # Grandparent OK
            ],
            # Check parent=/catalog/items (missing), check siblings, 456 fails load.
            # Loop continues to parent=/catalog (found).
            slugify_url("https://example.com/catalog"),
            {"data": "grandparent"},
            "source",
            slugify_url("https://example.com/catalog"),
        ),
        # 26. User origin is preserved (sibling match)
        (
            "https://example.com/user/followers",
            [
                (
                    slugify_url("https://example.com/user/verified_followers"),
                    {"data": "sibling"},
                    False,
                    "user",
                )
            ],  # Set origin='user'
            slugify_url("https://example.com/user/verified_followers"),
            {"data": "sibling"},
            "user",
            slugify_url("https://example.com/user/verified_followers"),
        ),
        # 27. User origin is preserved (parent match)
        (
            "https://example.com/user/followers",
            [
                (slugify_url("https://example.com/user"), {"data": "parent"}, False, "user")
            ],  # Set origin='user'
            slugify_url("https://example.com/user"),
            {"data": "parent"},
            "user",
            slugify_url("https://example.com/user"),
        ),
        # Test 28 removed as Path object simulation needs more work
    ],
)
def test_find_fallback_parser(
    target_url: str,
    available_parsers_setup: list[
        tuple[str, dict | None, bool] | tuple[str, dict | None, bool, ParserOrigin]
    ],  # Added origin tuple possibility
    expected_slug_key: str | None,
    expected_data: dict | None,
    expected_origin: ParserOrigin | None,
    expected_path_part: str | None,
):
    """Tests the find_fallback_parser utility with various fallback scenarios including siblings."""
    # Create the dict mapping slugs to ParserInfo tuples using the helper
    available_parsers_info: dict[str, ParserInfo] = {}
    for setup_tuple in available_parsers_setup:
        slug = setup_tuple[0]
        content = setup_tuple[1]
        fail_load = setup_tuple[2]
        origin: ParserOrigin = "source"  # Default origin
        if len(setup_tuple) == 4:
            # Ensure the 4th element is correctly typed if it exists
            if isinstance(setup_tuple[3], str) and setup_tuple[3] in ("source", "user"):
                origin = setup_tuple[3]
        available_parsers_info[slug] = create_mock_parser_info(slug, content, origin, fail_load)

    # Call the function
    result_candidates = find_fallback_parser(target_url, available_parsers_info)

    # Assert the results
    if expected_data is None:
        # If no data is expected, the candidate list should be empty
        assert not result_candidates, f"Expected no candidates, but got {len(result_candidates)}"
    else:
        # If data is expected, the list should not be empty
        assert result_candidates, (
            f"Expected candidates, but got an empty list for URL '{target_url}'"
        )

        # The tests expect the *first* candidate in the preference order
        # Unpack the first candidate tuple (it now has 4 elements)
        first_candidate = result_candidates[0]
        found_data: dict[str, Any]
        found_origin: ParserOrigin
        found_path: Path
        found_matched_slug: str
        found_data, found_origin, found_path, found_matched_slug = first_candidate

        # Assertions remain mostly the same, but check against the first candidate
        assert found_data == expected_data
        assert found_origin == expected_origin
        assert isinstance(found_path, Path)
        # Check that the path contains the expected slug key (as constructed by helper)
        assert expected_path_part is not None  # Should always be set if expected_data is not None
        assert expected_path_part in found_path.name

        # Verify that the *matched slug* from the candidate matches the expected slug key
        assert found_matched_slug == expected_slug_key, (
            f"Expected matched slug '{expected_slug_key}' but got '{found_matched_slug}'"
        )

        # Verify that the correct mock resource's read_text method was called ONCE
        # (The resource is the second element in the ParserInfo tuple)
        if expected_slug_key is not None:
            expected_parser_info = available_parsers_info.get(expected_slug_key)
            assert expected_parser_info is not None, (
                f"Test setup error: expected slug '{expected_slug_key}' not found."
            )
            try:
                # Access resource via index [1]
                expected_parser_info[1].read_text.assert_called_once()  # type: ignore
            except AssertionError as e:
                # Provide more context on failure
                # Access resource via index [1]
                calls = expected_parser_info[1].read_text.call_args_list  # type: ignore
                pytest.fail(f"AssertionError for slug '{expected_slug_key}': {e}\nCalls: {calls}")
