"""Quick and dirty reusable utility for types."""

def non_zero_uint(value):
    """Asserts that the value is an unsigned integer and not equal to zero."""

    iv = int(value)
    if iv <= 0:
        raise ValueError(f"value must be a non-zero unsigned integer, got {value}")

    return iv
