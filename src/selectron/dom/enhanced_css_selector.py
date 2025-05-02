import re

from selectron.dom.dom_views import DOMElementNode


def convert_simple_xpath_to_css_selector(xpath: str) -> str:
    """Converts simple XPath expressions to CSS selectors."""
    if not xpath:
        return ""

    # Remove leading slash if present
    xpath = xpath.lstrip("/")

    # Split into parts
    parts = xpath.split("/")
    css_parts = []

    for part in parts:
        if not part:
            continue

        # Handle custom elements with colons by escaping them
        if ":" in part and "[" not in part:
            base_part = part.replace(":", r"\:")
            css_parts.append(base_part)
            continue

        # Handle index notation [n]
        if "[" in part:
            base_part = part[: part.find("[")]
            # Handle custom elements with colons in the base part
            if ":" in base_part:
                base_part = base_part.replace(":", r"\:")
            index_part = part[part.find("[") :]

            # Handle multiple indices
            indices = [i.strip("[]") for i in index_part.split("]")[:-1]]

            for idx in indices:
                try:
                    # Handle numeric indices
                    if idx.isdigit():
                        index = int(idx) - 1
                        base_part += f":nth-of-type({index + 1})"
                    # Handle last() function
                    elif idx == "last()":
                        base_part += ":last-of-type"
                    # Handle position() functions
                    elif "position()" in idx:
                        if ">1" in idx:
                            base_part += ":nth-of-type(n+2)"
                except ValueError:
                    continue

            css_parts.append(base_part)
        else:
            css_parts.append(part)

    base_selector = " > ".join(css_parts)
    return base_selector


def enhanced_css_selector_for_element(
    element: DOMElementNode, include_dynamic_attributes: bool = True
) -> str:
    """
    Creates a CSS selector for a DOM element, handling various edge cases and special characters.

    Args:
            element: The DOM element to create a selector for

    Returns:
            A valid CSS selector string
    """
    try:
        # Get base selector from XPath
        css_selector = convert_simple_xpath_to_css_selector(element.xpath)

        # Handle class attributes
        if (
            "class" in element.attributes
            and element.attributes["class"]
            and include_dynamic_attributes
        ):
            # Define a regex pattern for valid class names in CSS
            valid_class_name_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]*$")

            # Iterate through the class attribute values
            classes = element.attributes["class"].split()
            for class_name in classes:
                # Skip empty class names
                if not class_name.strip():
                    continue

                # Check if the class name is valid
                if valid_class_name_pattern.match(class_name):
                    # Append the valid class name to the CSS selector
                    css_selector += f".{class_name}"
                else:
                    # Skip invalid class names
                    continue

        # Expanded set of safe attributes that are stable and useful for selection
        SAFE_ATTRIBUTES = {
            # Data attributes (if they're stable in your application)
            "id",
            # Standard HTML attributes
            "name",
            "type",
            "placeholder",
            # Accessibility attributes
            "aria-label",
            "aria-labelledby",
            "aria-describedby",
            "role",
            # Common form attributes
            "for",
            "autocomplete",
            "required",
            "readonly",
            # Media attributes
            "alt",
            "title",
            "src",
            # Custom stable attributes (add any application-specific ones)
            "href",
            "target",
        }

        if include_dynamic_attributes:
            dynamic_attributes = {
                "data-id",
                "data-qa",
                "data-cy",
                "data-testid",
            }
            SAFE_ATTRIBUTES.update(dynamic_attributes)

        # Handle other attributes
        for attribute, value in element.attributes.items():
            if attribute == "class":
                continue

            # Skip invalid attribute names
            if not attribute.strip():
                continue

            if attribute not in SAFE_ATTRIBUTES:
                continue

            # Escape special characters in attribute names
            safe_attribute = attribute.replace(":", r"\:")

            # Handle different value cases
            if value == "":
                css_selector += f"[{safe_attribute}]"
            elif any(char in value for char in "\"'<>`\n\r\t"):
                # Use contains for values with special characters
                # For newline-containing text, only use the part before the newline
                if "\n" in value:
                    value = value.split("\n")[0]
                # Regex-substitute *any* whitespace with a single space, then strip.
                collapsed_value = re.sub(r"\s+", " ", value).strip()
                # Escape embedded double-quotes.
                safe_value = collapsed_value.replace('"', '\\"')
                css_selector += f'[{safe_attribute}*="{safe_value}"]'
            else:
                css_selector += f'[{safe_attribute}="{value}"]'

        return css_selector

    except Exception:
        # Fallback to a more basic selector if something goes wrong
        tag_name = element.tag_name or "*"
        return f"{tag_name}[highlight_index='{element.highlight_index}']"
