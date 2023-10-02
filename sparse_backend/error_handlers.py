class InvalidIndexError(Exception):
    """Raised when an index is invalid"""


def validate_index(idx, max_val):
    """Validates that an index is valid"""
    if idx < 0 or idx >= max_val:
        raise InvalidIndexError(
            f"Invalid index. Must be between 0 and {max_val-1} inclusive."
        )


def handle_invalid_index(error):
    """Handles InvalidIndexError"""
    return str(error), 400
