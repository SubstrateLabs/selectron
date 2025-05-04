import re
import urllib.parse

from selectron.util.logger import get_logger

logger = get_logger(__name__)

# RFC 3986 unreserved characters (minus tilde ~)
# Tilde is excluded for stricter slug idempotency regarding the '~~' marker.
UNRESERVED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._"
# Precompute encoded index.html for efficiency (used in Step 7)
INDEX_HTML_ENCODED = "~~2findex.html"


def slugify_url(url: str) -> str:
    """
    Transforms a URL into a file-system-safe, reversible slug.

    This function applies several normalization and cleaning steps before encoding
    characters unsuitable for filenames into a '~~XX' format (where XX is the
    hex representation of the character code).

    The process aims for robustness and reasonable idempotency:
    1.  **Initial Cleaning:** Removes scheme (http/https), 'www.', lowercases host,
        strips trailing dot from host. Case-insensitive matching is used.
    2.  **Query/Fragment Removal:** Strips query parameters and fragment identifiers.
    3.  **Slash Normalization:** Collapses multiple consecutive slashes into one
        *after* scheme/host processing.
    4.  **Edge Case Handling:** Directly returns values for '/' (-> "~~2f") or
        empty results (e.g., from '#hash') to avoid issues in subsequent steps.
    5.  **Percent-Encoding Normalization:** Decodes all standard %XX sequences.
        This ensures URLs differing only by percent-encoding of unreserved
        characters (like %2D vs -) produce the same slug. Handles invalid sequences.
    6.  **~~XX Encoding:** Iterates through the cleaned, decoded URL. Unreserved
        characters (alphanumeric, -, ., _) are kept as-is. All other characters
        (including reserved chars like /, :, @, +, space, and previously decoded
        percent signs %) are encoded into the `~~XX` format using UTF-8 bytes.
    7.  **Post-Encoding Cleanup:** Removes specific trailing sequences from the
        *encoded* slug: trailing '~~2f' (encoded /) and trailing
        '~~2findex.html' (encoded /index.html).

    Args:
        url: The input URL string.

    Returns:
        A file-system-safe, reversible slug representing the normalized URL.
    """
    # === Step 1: Initial Cleaning ===
    cleaned_url = url
    # Use re.IGNORECASE for scheme and www. removal
    cleaned_url = re.sub(r"^https?:\/\/", "", cleaned_url, flags=re.IGNORECASE)
    cleaned_url = re.sub(r"^www\.", "", cleaned_url, flags=re.IGNORECASE)

    # Separate host/authority from the rest *before* slash normalization
    # to handle cases like `ftp://user@host//path` correctly.
    # We also handle protocol-relative URLs `//host/path`
    authority_part = ""
    path_part = cleaned_url  # Assume everything is path initially
    if "//" in cleaned_url:
        # May start with // (protocol relative) or have // after scheme removal
        if cleaned_url.startswith("//"):
            parts = cleaned_url[2:].split("/", 1)
            authority_part = parts[0]
            if len(parts) > 1:
                path_part = "/" + parts[1]
            else:
                path_part = ""  # Only authority like //example.com
        else:
            # Scheme was present, look for first / after potential authority
            parts = cleaned_url.split("/", 1)
            authority_part = parts[0]
            if len(parts) > 1:
                path_part = "/" + parts[1]
            else:
                path_part = ""  # Only authority like http://example.com
    elif "/" in cleaned_url:
        # Case like 'example.com/path' (no scheme/authority marker)
        parts = cleaned_url.split("/", 1)
        authority_part = parts[0]
        path_part = "/" + parts[1]
    else:
        # No slashes, assume it's all authority/host
        authority_part = cleaned_url
        path_part = ""

    # Handle IDN in authority: Encode host to Punycode (ASCII), then lowercase
    # Also strip trailing dots from the host part of the authority
    host = authority_part  # Assume authority is just host initially
    userinfo = ""
    port = ""
    if "@" in authority_part:
        userinfo, host = authority_part.split("@", 1)
        userinfo += "@"  # Keep the separator

    # Handle IPv6 literal address brackets for port/trailing dot logic
    ipv6_match = re.match(r"(\[.*?\])(:.*)?$", host)
    if ipv6_match:
        host_only = ipv6_match.group(1)
        port_maybe = ipv6_match.group(2) or ""
        host = host_only  # Keep brackets for IDNA check if needed
        port = port_maybe
    elif ":" in host:
        # Check for port on non-IPv6 host
        host_maybe, port_maybe = host.rsplit(":", 1)
        # Ensure it's likely a port number, not part of hostname like in IPv6
        if port_maybe.isdigit():
            host = host_maybe
            port = ":" + port_maybe
        # else: treat colon as part of host (though unusual)

    # Strip trailing dot from host part (before IDNA)
    host = host.rstrip(".")

    # Apply IDNA encoding/lowercasing
    try:
        host = host.encode("idna").decode("ascii").lower()
    except UnicodeError:
        logger.warning(
            f"NOTE: Host part '{host}' contained invalid characters for IDNA encoding; performing simple lowercase.",
            exc_info=True,
        )
        host = host.lower()

    # Reassemble authority
    authority_part = userinfo + host + port

    # Reassemble cleaned URL before final steps
    cleaned_url = authority_part + path_part

    # === Step 2: Query/Fragment Removal ===
    cleaned_url = cleaned_url.split("?")[0].split("#")[0]

    # === Step 3: Slash Normalization (Applied AFTER query/fragment removal) ===
    # We want to collapse runs of '/' within the *path* component while preserving:
    #   1) the double slash following an explicit scheme (e.g., "ftp://")
    #   2) protocol-relative leading "//" forms (e.g., "//example.com/path")

    scheme_match = re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", cleaned_url)
    if scheme_match:
        # Preserve the scheme and its following '//'
        prefix_len = scheme_match.end()  # index right after '://'
        prefix = cleaned_url[:prefix_len]
        rest = cleaned_url[prefix_len:]

        if "/" in rest:
            authority, path_after = rest.split("/", 1)
            normalized_path = "/" + re.sub(r"/+", "/", path_after.lstrip("/"))
            cleaned_url = prefix + authority + normalized_path
        else:
            cleaned_url = cleaned_url  # No path part, leave as-is
    else:
        # Handle protocol-relative URLs (starting with //)
        if cleaned_url.startswith("//"):
            prefix = "//"
            rest = cleaned_url[2:]
            if "/" in rest:
                authority, path_after = rest.split("/", 1)
                normalized_path = "/" + re.sub(r"/+", "/", path_after.lstrip("/"))
                cleaned_url = prefix + authority + normalized_path
            else:
                cleaned_url = prefix + rest  # No path
        else:
            # Typical host/path without explicit scheme
            if "/" in cleaned_url:
                authority, path_after = cleaned_url.split("/", 1)
                normalized_path = "/" + re.sub(r"/+", "/", path_after.lstrip("/"))
                cleaned_url = authority + normalized_path
            # else: no path – nothing to normalize

    # Convert multiple leading slashes (e.g., "///") down to a single '/'
    if cleaned_url and re.fullmatch(r"/+", cleaned_url):
        cleaned_url = "/"

    # === Step 4: Edge Case Handling ===
    if cleaned_url == "/":
        return "~~2f"
    # Handle truly empty result (e.g., only query/# removed entire content)
    if not cleaned_url:
        return ""

    # === Step 5: Percent-Encoding Normalization ===
    # Use 'surrogatepass' to handle potential lone surrogates from bad input
    try:
        decoded_url = urllib.parse.unquote(cleaned_url, errors="surrogatepass")
    except UnicodeDecodeError:
        logger.warning(
            f"NOTE: URL component '{cleaned_url}' contained undecodable percent-sequences; proceeding with potentially lossy decoding.",
            exc_info=True,
        )
        # Attempt replacement on error - best effort
        decoded_url = urllib.parse.unquote(cleaned_url, errors="replace")

    # === Step 6: ~~XX Encoding ===
    result_bytes = bytearray()
    for char in decoded_url:
        if char in UNRESERVED_CHARS:
            # Optimization: Use ord() for known ASCII chars
            result_bytes.append(ord(char))
        else:
            # Encode character to UTF-8 bytes and append ~~XX for each byte
            # Use 'surrogatepass' to handle potential lone surrogates if unquote produced them
            utf8_bytes = char.encode("utf-8", errors="surrogatepass")
            for byte_val in utf8_bytes:
                result_bytes.extend(f"~~{format(byte_val, '02x')}".encode("ascii"))

    # Convert final bytearray result to string
    result = result_bytes.decode("ascii")  # Should always be valid ASCII

    # === Step 7: Post-Encoding Cleanup ===
    # Remove trailing slash only if it wasn't the *only* character
    if result.endswith("~~2f") and result != "~~2f":
        result = result[:-4]
    # Use precomputed constant
    if result.endswith(INDEX_HTML_ENCODED):
        result = result[: -len(INDEX_HTML_ENCODED)]

    return result


def unslugify_url(slug: str) -> str:
    """Reverses the slugification process, decoding ~~XX sequences back into characters (handling UTF-8)."""
    # Use a regex to find all ~~XX sequences, case-insensitive hex
    encoded_parts = re.split(r"(~~[0-9a-fA-F]{2})", slug)

    result_bytes = bytearray()
    for part in encoded_parts:
        if not part:
            continue  # Skip empty strings from split
        if part.startswith("~~") and len(part) == 4:
            try:
                hex_val = part[2:]
                byte_val = int(hex_val, 16)
                # Optional: Log if we are decoding control characters
                if 0 <= byte_val < 32 or byte_val == 127:
                    logger.debug(
                        f"NOTE: Decoded ASCII control character {hex(byte_val)} from slug part '{part}'"
                    )
                result_bytes.append(byte_val)
            except ValueError:
                # Should not happen with the stricter regex, but belts and suspenders
                logger.warning(
                    f"NOTE: Encountered invalid hex sequence '{part}' during unslugify; treating literally."
                )
                result_bytes.extend(part.encode("utf-8", errors="replace"))
        else:
            # Literal part, encode directly to bytes
            result_bytes.extend(part.encode("utf-8", errors="replace"))

    # Decode the assembled bytes using UTF-8, replacing errors
    try:
        return result_bytes.decode("utf-8", errors="strict")  # Be strict on final decode
    except UnicodeDecodeError:
        logger.warning(
            "NOTE: Final byte sequence from unslugify contained invalid UTF-8; replacing errors.",
            exc_info=True,
        )
        return result_bytes.decode("utf-8", errors="replace")
