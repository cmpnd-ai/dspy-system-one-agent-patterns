"""Parse a small TOML-like config format into nested dictionaries."""


def _coerce(value: str):
    """Turn a raw value string into a bool, int, float, or str."""
    if value in ("true", "false"):
        return value == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip('"')


def parse_config(text: str) -> dict:
    """Parse `key = value` lines grouped under `[section]` or `[section.sub]` headers."""
    result: dict = {}
    current = result
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            parts = line[1:-1].split(".")
            current = result.setdefault(parts[0], {})
            for part in parts[2:]:
                current = current.setdefault(part, {})
            continue
        key, _, value = line.partition("=")
        current[key.strip()] = _coerce(value.strip())
    return result
