import re


def slugify_url(url: str) -> str:
    # 1. Basic Cleaning
    url = re.sub(r"^https?:\/\/", "", url)  # Remove http/https
    url = re.sub(r"^www\.", "", url)  # Remove www.
    url = url.split("?")[0].split("#")[0]  # Remove query/hash
    url = re.sub(r"/+$", "", url)  # Remove trailing slashes
    url = re.sub(r"/index\.html$", "", url)  # Remove index.html
    url = re.sub(r"/+", "/", url)  # Normalize slashes

    # 2. Simple Encoding (ASCII only)
    result = ""
    for char in url:
        if re.match(r"[a-zA-Z0-9-]", char):
            result += char
        elif char == ".":
            result += "~~2e"
        elif char == "/":
            result += "~~2f"
        elif char == ":":
            result += "~~3a"
        elif char == "@":
            result += "~~40"
        elif char == " ":
            result += "~~20"
        # Ignore other non-alphanumeric ASCII chars for simplicity, or encode them?
        # Let's encode any remaining non-alphanumeric ASCII to be safe.
        elif 32 < ord(char) < 127:  # Printable ASCII other than those above
            result += "~~" + hex(ord(char))[2:].lower()
        # Ignore non-printable ASCII or non-ASCII entirely

    return result


def unslugify_url(slug: str) -> str:
    def replace_match(match: re.Match[str]) -> str:
        hex_val = match.group(1)
        # Ensure hex_val is exactly 2 digits before attempting conversion
        if len(hex_val) == 2:
            try:
                return chr(int(hex_val, 16))
            except ValueError:
                # If not valid hex, return the original sequence
                return f"~~{hex_val}"
        else:
            # If not 2 digits (e.g., from previous complex encoding), return original
            return f"~~{hex_val}"

    # Only replace ~~ followed by exactly 2 hex digits
    return re.sub(r"~~([0-9a-f]{2})", replace_match, slug)
