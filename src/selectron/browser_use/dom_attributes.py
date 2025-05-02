# Define the list of attributes we want to see in the .dom.txt output
# These help identify elements more robustly.
DOM_STRING_INCLUDE_ATTRIBUTES = [
    "id",
    "class",
    "name",
    "role",
    "aria-label",
    "aria-labelledby",
    "aria-describedby",
    "placeholder",
    "title",
    "alt",
    "href",
    "type",
    "value",  # Useful for input fields
    "for",  # Useful for labels
    "data-testid",
    "data-cy",
    "data-qa",
    # Add other relevant attributes like 'pattern', 'required', 'disabled', etc. if needed
]
