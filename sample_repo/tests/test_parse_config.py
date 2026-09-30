from config_loader import parse_config


def make_config(*lines: str) -> str:
    """Build config text from lines, the way every test fixture here does."""
    return "\n".join(lines) + "\n"


def test_parses_flat_keys():
    text = make_config("name = demo", "retries = 3", "ratio = 0.5", "debug = true")
    assert parse_config(text) == {"name": "demo", "retries": 3, "ratio": 0.5, "debug": True}


def test_parses_sections():
    text = make_config("[server]", "host = localhost", "port = 8080")
    assert parse_config(text) == {"server": {"host": "localhost", "port": 8080}}


def test_parses_nested_sections():
    text = make_config("[database]", "name = app", "[database.pool]", "size = 5", "timeout = 2.5")
    assert parse_config(text) == {"database": {"name": "app", "pool": {"size": 5, "timeout": 2.5}}}
