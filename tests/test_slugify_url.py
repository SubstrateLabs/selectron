import re
import urllib.parse  # Needed for comparison in roundtrip test

import pytest

from selectron.util.slugify_url import slugify_url, unslugify_url

# RFC 3986 unreserved characters (minus tilde ~) for comparison logic
# Tilde (~) is excluded to ensure that the slug encoding marker '~~' itself
# gets encoded if it appears in the original URL, improving idempotency.
UNRESERVED_CHARS_RE = re.compile(r"[a-zA-Z0-9-._]")


@pytest.mark.parametrize(
    "original_url, expected_slug",
    [
        # Basic cleaning and encoding (unchanged expectations)
        ("https://www.example.com/path?query=1#hash", "example.com~~2fpath"),
        (
            "http://EXAMPLE.com/another/path/",  # Uppercase host
            "example.com~~2fanother~~2fpath",
        ),  # trailing slash removed, host lowercased
        ("example.com/index.html", "example.com"),
        ("example.com//double//slash", "example.com~~2fdouble~~2fslash"),  # slash normalization
        # Ports
        ("example.com:8080/path", "example.com~~3a8080~~2fpath"),
        ("example.com:80", "example.com~~3a80"),
        # Path Case Sensitivity
        ("example.com/PATH/TO/RESOURCE", "example.com~~2fPATH~~2fTO~~2fRESOURCE"),
        # Encoding of reserved chars
        ("user:pass@example.com", "user~~3apass~~40example.com"),
        ("example.com/file name", "example.com~~2ffile~~20name"),  # Space
        ("example.com/path+plus", "example.com~~2fpath~~2bplus"),  # Plus
        ("example.com/path%2Fslash", "example.com~~2fpath~~2fslash"),  # Encoded slash handled
        ("example.com/path%20space", "example.com~~2fpath~~20space"),  # Encoded space handled
        ("example.com/star*path", "example.com~~2fstar~~2apath"),  # Asterisk
        # Path Segments
        ("example.com/./path", "example.com~~2f.~~2fpath"),  # . is unreserved
        ("example.com/../path", "example.com~~2f..~~2fpath"),  # . is unreserved
        # Percent-encoding normalization (unreserved chars)
        ("example.com/path%2Dhyphen", "example.com~~2fpath-hyphen"),  # %2D -> -
        ("example.com/path%5Funderscore", "example.com~~2fpath_underscore"),  # %5F -> _
        ("example.com/path%2Eperiod", "example.com~~2fpath.period"),  # %2E -> .
        ("example.com/path%7Etilde", "example.com~~2fpath~~7etilde"),  # %7E -> ~ -> ~~7e
        # Mixed-case percent encoding
        ("example.com/path%2fslash", "example.com~~2fpath~~2fslash"),  # lowercase %2f
        ("example.com/path%2Fslash", "example.com~~2fpath~~2fslash"),  # uppercase %2F
        # Mixed / Other
        (
            "Example.com/path_with%2Dstuff.html",
            "example.com~~2fpath_with-stuff.html",
        ),  # Mixed case host, underscore, hyphen encoding
        (
            "example.com/a%2Fb%2Fc",
            "example.com~~2fa~~2fb~~2fc",
        ),  # Decodes %2F then re-encodes as ~~2f
        # Decodes %25 to %, then processes literal %, 2, F, b
        (
            "example.com/a%252Fb",
            "example.com~~2fa~~252Fb",
        ),  # %25->%, then encode % -> ~~25, keep literal 2, F, b (unreserved)
        # Idempotency (slugging an already slugged-like string)
        (
            "example.com~~2fpath",  # Contains our encoding marker
            "example.com~~7e~~7e2fpath",  # ~ gets encoded to ~~7e
        ),
        # Edge cases
        ("/", "~~2f"),  # Only root path
        ("#hash", ""),  # Only fragment
        ("?query=val", ""),  # Only query
        ("http://", ""),  # Scheme only
        ("http://example.com", "example.com"),  # Scheme and host only
        ("www.", ""),  # Only www.
        ("", ""),  # Empty string
        # IPv6 Address Handling
        ("http://[::1]/path", "~~5b~~3a~~3a1~~5d~~2fpath"),  # Basic IPv6
        (
            "http://[2001:db8::1]:8080/p",
            "~~5b2001~~3adb8~~3a~~3a1~~5d~~3a8080~~2fp",
        ),  # IPv6 with port
        # Authority-Only URLs
        ("http://user:pass@", "user~~3apass~~40"),
        ("http://[::1]", "~~5b~~3a~~3a1~~5d"),
        # Invalid Percent Encoding
        (
            "example.com/test%AXok",
            "example.com~~2ftest~~25AXok",
        ),  # % is encoded, A and X are kept (case preserved)
        # Unicode Characters
        (
            "example.com/你好世界",
            "example.com~~2f~~e4~~bd~~a0~~e5~~a5~~bd~~e4~~b8~~96~~e7~~95~~8c",
        ),  # Chinese
        ("example.com/file-with-é", "example.com~~2ffile-with-~~c3~~a9"),  # French accent
        # Unicode in host (punycoded) and path (~~ encoded)
        ("héllö.com/wörld", "xn--hll-bma1e.com~~2fw~~c3~~b6rld"),
        # Internationalized Domain Names (IDN)
        ("例子.com/path", "xn--fsqu00a.com~~2fpath"),  # Unicode host converted to punycode
        ("xn--fsqu00a.com/path", "xn--fsqu00a.com~~2fpath"),  # Punycode host
        ("XN--FSQU00A.COM/path", "xn--fsqu00a.com~~2fpath"),  # Uppercase Punycode host
        # Control Characters
        (
            "example.com/path%0Awith%09tabs",
            "example.com~~2fpath~~0awith~~09tabs",
        ),  # Newline and tab encoded
        # Punycode-like Path
        (
            "example.com/xn--fsqu00a-in-path",
            "example.com~~2fxn--fsqu00a-in-path",
        ),  # Dashes are unreserved
        # New test cases from review
        ("HTTPS://WWW.EXAMPLE.COM/", "example.com"),  # Case insensitive scheme/www
        ("//example.com/path", "example.com~~2fpath"),  # Protocol-relative
        ("ftp://example.com/p", "ftp~~3a~~2f~~2fexample.com~~2fp"),  # Non-http scheme kept
        ("example.com/Index.html", "example.com~~2fIndex.html"),  # Mixed-case index not stripped
        ("example.com/index.htm", "example.com~~2findex.htm"),  # .htm not stripped
        ("example.com/abc%ZZdef", "example.com~~2fabc~~25ZZdef"),  # Invalid %XX
        ("example.com/abc%2", "example.com~~2fabc~~252"),  # Truncated %X
        ("user@example.com:8080/p", "user~~40example.com~~3a8080~~2fp"),  # Userinfo + port
        (
            "example.com~~2Ffile",
            "example.com~~7e~~7e2ffile",
        ),  # Uppercase hex in slug marker encoded (becomes lowercase)
        ("example.com/😀", "example.com~~2f~~f0~~9f~~98~~80"),  # Emoji
        ("example.com.", "example.com"),  # Trailing dot on host
        ("http://example.com./", "example.com"),  # Trailing dot + slash
        ("http://example.com//a///b", "example.com~~2fa~~2fb"),  # Multi-slash after host
        ("//host//path", "host~~2fpath"),  # Multi-slash in protocol-relative
        ("http://[fe80::1%eth0]/p", "~~5bfe80~~3a~~3a1~~25eth0~~5d~~2fp"),  # IPv6 scope ID
    ],
)
def test_slugify_url_normalized(original_url, expected_slug):
    assert slugify_url(original_url) == expected_slug


# Test the round trip: normalized slugify -> simple unslugify
@pytest.mark.parametrize(
    "original_url",
    [
        "https://www.example.com/path?query=1#hash",
        "http://EXAMPLE.com/another/path/",
        "example.com/file name",
        "user:pass@example.com/path%20with%20space%2Dand%2Echars",
        "example.com/path_plus+tilde~chars.test",
        "example.com/a%2Fb%2Fc",
        "example.com/a%252Fb",  # Encoded percent
        # Add some new edge cases to roundtrip
        "example.com:8080/path",
        "example.com/./path",
        "/",
        "http://example.com",
        # Add IPv6 and Path Case examples
        "http://[::1]/path",
        "example.com/PATH/TO/RESOURCE",
        "example.com/test%AXok",
        # Add Unicode/IDN examples
        "example.com/你好世界",
        "héllö.com/wörld",
        "例子.com/path",
        "XN--FSQU00A.COM/path",
        # Add control char / punycode-path cases
        "example.com/path%0Awith%09tabs",
        "example.com/xn--fsqu00a-in-path",
        # Add new review cases for roundtrip
        "HTTPS://WWW.EXAMPLE.COM/",
        "//example.com/path",
        "ftp://example.com/p",
        "example.com/Index.html",
        "example.com/abc%ZZdef",
        "example.com/abc%2",
        "user@example.com:8080/p",
        "example.com~~2Ffile",
        "example.com/😀",
        "example.com.",
        "http://example.com./",
        "http://example.com//a///b",
        "//host//path",
        "http://[fe80::1%eth0]/p",
    ],
)
def test_normalized_slugify_unslugify_roundtrip(original_url):
    # 1. Calculate the expected result after cleaning and %-decoding.
    #    This precisely mirrors the state of the URL within slugify_url
    #    *just before* it performs the final ~~XX re-encoding loop,
    #    *and* before the final trailing / and /index.html cleanup.
    #    unslugify_url only reverses the ~~XX encoding, not the cleanup steps.

    # Mimic Step 1: Initial Cleaning (case-insensitive scheme/www)
    expected = original_url
    expected = re.sub(r"^https?:\/\/", "", expected, flags=re.IGNORECASE)
    expected = re.sub(r"^www\.", "", expected, flags=re.IGNORECASE)

    # Mimic Step 1: Separate authority/path carefully
    authority_part = ""
    path_part = expected  # Assume all path initially
    if "//" in expected:
        if expected.startswith("//"):  # Protocol-relative
            parts = expected[2:].split("/", 1)
            authority_part = parts[0]
            path_part = "/" + parts[1] if len(parts) > 1 else ""
        else:  # Scheme was present
            parts = expected.split("/", 1)
            authority_part = parts[0]
            path_part = "/" + parts[1] if len(parts) > 1 else ""
    elif "/" in expected:  # No scheme/authority marker, e.g., example.com/path
        parts = expected.split("/", 1)
        authority_part = parts[0]
        path_part = "/" + parts[1]
    else:  # No slashes, all authority
        authority_part = expected
        path_part = ""

    # Mimic Step 1: Clean authority (IDN, case, trailing dot, userinfo, port)
    host = authority_part
    userinfo = ""
    port = ""
    if "@" in authority_part:
        userinfo, host = authority_part.split("@", 1)
        userinfo += "@"
    ipv6_match = re.match(r"(\[.*?\])(:.*)?$", host)
    if ipv6_match:
        host_only = ipv6_match.group(1)
        port_maybe = ipv6_match.group(2) or ""
        host = host_only
        port = port_maybe
    elif ":" in host:
        host_maybe, port_maybe = host.rsplit(":", 1)
        if port_maybe.isdigit():
            host = host_maybe
            port = ":" + port_maybe
    host = host.rstrip(".")  # Strip trailing dot
    try:  # IDNA encode/decode/lowercase
        host = host.encode("idna").decode("ascii").lower()
    except UnicodeError:
        host = host.lower()
    authority_part = userinfo + host + port  # Reassemble authority

    # Reassemble cleaned URL before query/fragment/slash steps
    expected = authority_part + path_part

    # Mimic Step 2: Query/Fragment Removal
    expected = expected.split("?")[0].split("#")[0]

    # Mimic Step 3: Slash Normalization (applied after authority/query removal)
    # This logic needs to mirror the complex slash normalization in slugify_url
    scheme_match = re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", expected)
    if scheme_match:
        prefix_len = scheme_match.end()
        prefix = expected[:prefix_len]
        rest = expected[prefix_len:]
        if "/" in rest:
            auth, path_after = rest.split("/", 1)
            normalized_path = "/" + re.sub(r"/+", "/", path_after.lstrip("/"))
            expected = prefix + auth + normalized_path
        # else: no path after scheme+authority, leave as is
    elif expected.startswith("//"):  # Protocol-relative
        prefix = "//"
        rest = expected[2:]
        if "/" in rest:
            auth, path_after = rest.split("/", 1)
            normalized_path = "/" + re.sub(r"/+", "/", path_after.lstrip("/"))
            expected = prefix + auth + normalized_path
        # else: no path after //authority, leave as is
    elif "/" in expected:  # No scheme, simple authority/path
        auth, path_after = expected.split("/", 1)
        normalized_path = "/" + re.sub(r"/+", "/", path_after.lstrip("/"))
        expected = auth + normalized_path
    # else: no slashes at all, nothing to normalize

    # Handle /// -> / edge case
    if expected and re.fullmatch(r"/+", expected):
        expected = "/"

    # Mimic the *effect* of Step 7's trailing slash removal on the pre-encoded state
    if expected.endswith("/") and expected != "/":
        expected = expected[:-1]

    # Handle root / path early exit from slugify *before* decode/encode
    if expected == "/":
        # The expected result after unslugify("~~2f") is "/"
        pass  # Keep expected = "/" for the final comparison
    elif not expected:
        # The expected result after unslugify("") is ""
        pass  # Keep expected = ""
    else:
        # Mimic step 5: Final decode before ~~XX encoding would happen
        # This is the state unslugify should return to.
        # We do NOT mimic step 7 (post-encoding cleanup) here.
        expected = urllib.parse.unquote(expected, errors="surrogatepass")

    # 2. Perform the actual round trip using the functions
    slug = slugify_url(original_url)
    unslugged = unslugify_url(slug)

    # 3. Compare the result of unslugify(slugify(url)) with our carefully
    #    calculated expected state.
    assert unslugged == expected


# Test that unslugify hasn't changed behavior for its original simple cases
@pytest.mark.parametrize(
    "slug, expected_url",
    [
        ("example~~2ecom~~2fpath", "example.com/path"),
        ("user~~3apass~~40example~~2ecom", "user:pass@example.com"),
        ("example~~2ecom~~2fpath~~20with~~20space", "example.com/path with space"),
        ("example~~2ecom~~2fpath~~5fplus~~2btest", "example.com/path_plus+test"),
        # Cases that might be produced by the *new* slugify
        ("example.com~~2fpath-hyphen", "example.com/path-hyphen"),  # Contains unreserved chars
        ("example.com~~2fpath_underscore", "example.com/path_underscore"),
        ("example.com~~2fpath.period", "example.com/path.period"),
        ("example.com~~2fpath~tilde", "example.com/path~tilde"),
        (
            "example.com~~2fpath~~7etilde",
            "example.com/path~tilde",
        ),  # Test unslugify with encoded tilde
        # New review cases for unslugify
        ("ftp~~3a~~2f~~2fexample.com", "ftp://example.com"),  # Non-http scheme
        ("example.com~~2fabc~~7Edef", "example.com/abc~def"),  # Tilde (capital hex)
        ("example.com~~2f~~00nul", "example.com/\x00nul"),  # Control byte (nul)
        ("example.com~~7e~~7e2Ffile", "example.com~~2Ffile"),  # Encoded slug marker with upper hex
    ],
)
def test_unslugify_simple_cases(slug, expected_url):
    assert unslugify_url(slug) == expected_url
