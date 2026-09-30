"""Eight fake prior agent turns that every demo passes in as `history`.

The chunks come in three kinds so Pattern 3 (history relevance) has something to sort:
  - three short chunks that matter for the failing test ("show" is right, and cheap),
  - three medium chunks about unrelated work ("hide" is a visible saving),
  - two long chunks where one line matters ("summarize" earns its keep).
"""

# Relevant, short: an earlier search that found the parser.
_GREP_RESULT = """\
[turn 3] tool: search_files(pattern="def parse_config", path=".")
config_loader/parse_config.py:18:def parse_config(text: str) -> dict:
config_loader/__init__.py:1:from config_loader.parse_config import parse_config
tests/test_parse_config.py:1:from config_loader import parse_config
note: parse_config is the only parser in the package. Section headers are handled
in the `if line.startswith("[")` branch, which splits the header on "." and walks
the parts to find the dictionary that later `key = value` lines are written into."""

# Relevant, short: how the test fixtures are built.
_FIXTURE_NOTE = """\
[turn 4] note to self: tests/test_parse_config.py builds each fixture with
make_config(*lines), which joins the lines with "\\n" and adds a trailing newline.
There is no conftest fixture and no file I/O; every test calls parse_config(text)
directly and compares the whole returned dict with ==. So a failing test means the
dict shape is wrong, not that a file failed to load. The nested-section test uses
"[database]" followed by "[database.pool]", and it expects size and timeout inside a
"pool" dictionary under "database", next to the "name" key."""

# Relevant, short: the user's earlier description of the config format.
_FORMAT_NOTE = """\
[turn 1] user: The config format is a tiny TOML subset. `[a]` opens a section, and
`[a.b]` opens a sub-section nested inside `a`, so keys after `[a.b]` should end up
at result["a"]["b"][key]. Values can be ints, floats, true/false, or strings.
Sections can nest more than one level deep, so `[a.b.c]` should land keys at
result["a"]["b"]["c"][key]. Opening `[a]` again later should return to the same
dictionary rather than replacing it, and a sub-section should never overwrite the keys
that already sit directly under its parent section."""

# Irrelevant, medium: a discussion about the README.
_README_DISCUSSION = """\
[turn 2] user: Before we touch any code, can you look at the README? I think it
undersells the project. Right now it is two sentences. I'd like a short "Why" section
that explains we wrote config_loader because the full TOML libraries pulled in more
dependencies than we wanted for a small CLI, a "Format" section with one example
file, and a badge row at the top for CI status and the license.
assistant: Here is a draft outline for the README:
  1. Title and badge row (CI, license, Python versions 3.10 to 3.13).
  2. One-paragraph summary: config_loader reads a small TOML-like format into plain
     dictionaries with no third-party dependencies.
  3. "Why": the standard TOML parsers are excellent but heavier than this CLI needs,
     and we only use a handful of features: sections, sub-sections, and scalar values.
  4. "Format": an example with a [server] section, a [database] section, and a
     [database.pool] sub-section, with a note that arrays and inline tables are not
     supported on purpose.
  5. "Development": how to run the tests with pytest and how to cut a release.
user: Good. Hold off on writing it until the tests are green; we'll come back to it.
assistant: Understood. I'll leave the README alone for now and note the outline here
so we can pick it up later. I'd also suggest a CHANGELOG.md with a single "Unreleased"
heading so contributors have somewhere to record changes, but that can wait too."""

# Irrelevant, medium: a package-install log.
_INSTALL_LOG = """\
[turn 5] tool: run_shell_command(cmd="pip install -e '.[dev]'")
Obtaining file:///work/config_loader
  Installing build dependencies ... done
  Getting requirements to build editable ... done
  Preparing editable metadata (pyproject.toml) ... done
Collecting pytest>=8.0 (from config-loader==0.3.1)
  Using cached pytest-8.3.4-py3-none-any.whl.metadata (7.5 kB)
Collecting ruff>=0.6 (from config-loader==0.3.1)
  Downloading ruff-0.8.4-py3-none-macosx_11_0_arm64.whl.metadata (25 kB)
Collecting mypy>=1.10 (from config-loader==0.3.1)
  Using cached mypy-1.13.0-cp312-cp312-macosx_11_0_arm64.whl.metadata (2.1 kB)
Collecting iniconfig (from pytest>=8.0->config-loader==0.3.1)
  Using cached iniconfig-2.0.0-py3-none-any.whl.metadata (2.6 kB)
Collecting packaging (from pytest>=8.0->config-loader==0.3.1)
  Using cached packaging-24.2-py3-none-any.whl.metadata (3.2 kB)
Collecting pluggy<2,>=1.5 (from pytest>=8.0->config-loader==0.3.1)
  Using cached pluggy-1.5.0-py3-none-any.whl.metadata (4.8 kB)
Building wheels for collected packages: config-loader
  Building editable for config-loader (pyproject.toml) ... done
  Created wheel for config-loader: filename=config_loader-0.3.1-0.editable-py3-none-any.whl
Successfully built config-loader
Installing collected packages: typing-extensions, ruff, pluggy, packaging, mypy-extensions,
  iniconfig, pytest, mypy, config-loader
Successfully installed config-loader-0.3.1 iniconfig-2.0.0 mypy-1.13.0 mypy-extensions-1.0.0
  packaging-24.2 pluggy-1.5.0 pytest-8.3.4 ruff-0.8.4 typing-extensions-4.12.2
[notice] A new release of pip is available: 24.2 -> 24.3.1"""

# Irrelevant, medium: a CI configuration discussion.
_CI_DISCUSSION = """\
[turn 6] user: Separate question: our CI takes eleven minutes and most of it is setup.
Can we cache the pip downloads and run the lint job in parallel with the tests?
assistant: Yes. Three changes to .github/workflows/ci.yml would help:
  - Use actions/setup-python with `cache: pip` so the wheel downloads are reused
    between runs, keyed on the hash of pyproject.toml.
  - Split the single job into two jobs, `lint` (ruff check, ruff format --check, and
    mypy) and `test` (pytest on a matrix of Python 3.10, 3.11, 3.12, and 3.13). They
    don't depend on each other, so they can run at the same time.
  - Add `concurrency: { group: ci-${{ github.ref }}, cancel-in-progress: true }` so
    pushing twice to a branch cancels the stale run instead of queueing it.
I would expect the wall-clock time to drop to about four minutes, dominated by the
slowest matrix entry. We could also drop 3.10 from the matrix if nobody depends on it,
but that is a support decision rather than a CI one.
user: Let's do the first and third now and talk about the matrix later. Not today
though; the failing test comes first.
assistant: Noted. I'll write the CI changes up as a follow-up task with the cache key,
the job split, and the concurrency group, and leave the matrix question open."""

# Borderline, long: a full pytest run where only the assertion line matters.
_PYTEST_OUTPUT = """\
[turn 7] tool: run_shell_command(cmd="pytest -vv")
============================= test session starts ==============================
platform darwin -- Python 3.12.7, pytest-8.3.4, pluggy-1.5.0 -- /work/.venv/bin/python
cachedir: .pytest_cache
rootdir: /work/config_loader
configfile: pyproject.toml
plugins: cov-6.0.0, xdist-3.6.1, randomly-3.16.0
Using --randomly-seed=2871634402
collecting ... collected 3 items

tests/test_parse_config.py::test_parses_sections PASSED                  [ 33%]
tests/test_parse_config.py::test_parses_nested_sections FAILED           [ 66%]
tests/test_parse_config.py::test_parses_flat_keys PASSED                 [100%]

=================================== FAILURES ===================================
_________________________ test_parses_nested_sections __________________________

    def test_parses_nested_sections():
        text = make_config("[database]", "name = app", "[database.pool]", "size = 5", "timeout = 2.5")
>       assert parse_config(text) == {"database": {"name": "app", "pool": {"size": 5, "timeout": 2.5}}}
E       AssertionError: assert {'database': {'name': 'app', 'size': 5, 'timeout': 2.5}} == {'database': {'name': 'app', 'pool': {'size': 5, 'timeout': 2.5}}}
E
E         Differing items:
E         {'database': {'name': 'app', 'size': 5, 'timeout': 2.5}} != {'database': {'name': 'app', 'pool': {'size': 5, 'timeout': 2.5}}}
E
E         Full diff:
E           {
E               'database': {
E                   'name': 'app',
E         -         'pool': {
E         -             'size': 5,
E         -             'timeout': 2.5,
E         -         },
E         +         'size': 5,
E         +         'timeout': 2.5,
E               },
E           }

tests/test_parse_config.py:21: AssertionError
=============================== warnings summary ===============================
.venv/lib/python3.12/site-packages/_pytest/config/__init__.py:1441
  /work/.venv/lib/python3.12/site-packages/_pytest/config/__init__.py:1441:
  PytestConfigWarning: Unknown config option: asyncio_mode

    self._warn_or_fail_if_strict(f"Unknown config option: {key}\\n")

.venv/lib/python3.12/site-packages/pytest_randomly/__init__.py:218
  /work/.venv/lib/python3.12/site-packages/pytest_randomly/__init__.py:218:
  DeprecationWarning: pkg_resources is deprecated as an API. See
  https://setuptools.pypa.io/en/latest/pkg_resources.html

    import pkg_resources

.venv/lib/python3.12/site-packages/xdist/plugin.py:275
  /work/.venv/lib/python3.12/site-packages/xdist/plugin.py:275:
  PytestDeprecationWarning: The --rsyncdir command line argument and rsyncdirs
  config variable are deprecated. The rsync feature will be removed in pytest-xdist 4.0.

    config.issue_config_time_warning(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html

---------- coverage: platform darwin, python 3.12.7-final-0 ----------
Name                               Stmts   Miss Branch BrPart  Cover   Missing
------------------------------------------------------------------------------
config_loader/__init__.py              2      0      0      0   100%
config_loader/parse_config.py         24      2     12      2    89%   12-13, 29->30
tests/test_parse_config.py            12      0      0      0   100%
------------------------------------------------------------------------------
TOTAL                                 38      2     12      2    92%

---------- slowest 10 durations ----------
0.01s call     tests/test_parse_config.py::test_parses_nested_sections
0.00s setup    tests/test_parse_config.py::test_parses_sections
0.00s call     tests/test_parse_config.py::test_parses_sections
0.00s setup    tests/test_parse_config.py::test_parses_nested_sections
0.00s setup    tests/test_parse_config.py::test_parses_flat_keys
0.00s call     tests/test_parse_config.py::test_parses_flat_keys
0.00s teardown tests/test_parse_config.py::test_parses_sections
0.00s teardown tests/test_parse_config.py::test_parses_nested_sections
0.00s teardown tests/test_parse_config.py::test_parses_flat_keys
=========================== short test summary info ============================
FAILED tests/test_parse_config.py::test_parses_nested_sections - AssertionError: assert {'database': {'name': 'app', 'size': 5, 'timeout': 2.5}} == {'database': {'name': 'app', 'pool': {'size': 5, 'timeout': 2.5}}}
=================== 1 failed, 2 passed, 3 warnings in 0.09s ===================="""

# Borderline, long: a git log where one commit touched the section-header code.
_GIT_LOG = """\
[turn 8] tool: run_shell_command(cmd="git log --stat -n 12")
commit 9f3c2a1e4b7d8c6f5a0e9d2b1c3f4a5e6d7c8b9a
Author: Priya Raman <priya@example.com>
Date:   Mon Sep 22 16:41:07 2025 -0700

    Bump version to 0.3.1

 pyproject.toml | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)

commit 4e8d1b2c3a5f6e7d8c9b0a1f2e3d4c5b6a7f8e9d
Author: Marcus Lee <marcus@example.com>
Date:   Mon Sep 22 11:02:55 2025 -0700

    Add test for nested sections

 tests/test_parse_config.py | 5 +++++
 1 file changed, 5 insertions(+)

commit b1a2c3d4e5f60718293a4b5c6d7e8f9012345678
Author: Priya Raman <priya@example.com>
Date:   Fri Sep 19 17:20:13 2025 -0700

    Refactor section header parsing to reuse the top-level setdefault

 config_loader/parse_config.py | 9 ++++-----
 1 file changed, 4 insertions(+), 5 deletions(-)

commit 7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0d9c8b
Author: Dependabot <support@github.com>
Date:   Thu Sep 18 03:12:44 2025 +0000

    Bump ruff from 0.8.3 to 0.8.4

 pyproject.toml | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)

commit 2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e
Author: Marcus Lee <marcus@example.com>
Date:   Wed Sep 17 14:30:21 2025 -0700

    Coerce true/false to bool

 config_loader/parse_config.py | 2 ++
 tests/test_parse_config.py    | 4 ++--
 2 files changed, 4 insertions(+), 2 deletions(-)

commit 5f6e7d8c9b0a1f2e3d4c5b6a7f8e9d0c1b2a3f4e
Author: Priya Raman <priya@example.com>
Date:   Tue Sep 16 10:05:39 2025 -0700

    Skip blank lines and comments

 config_loader/parse_config.py | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)

commit 8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b
Author: Dependabot <support@github.com>
Date:   Mon Sep 15 03:10:02 2025 +0000

    Bump actions/checkout from 4.1.7 to 4.2.0

 .github/workflows/ci.yml | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)

commit 0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f
Author: Marcus Lee <marcus@example.com>
Date:   Fri Sep 12 15:48:30 2025 -0700

    Add CI workflow with lint and test steps

 .github/workflows/ci.yml | 31 +++++++++++++++++++++++++++++++
 1 file changed, 31 insertions(+)

commit 3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d
Author: Priya Raman <priya@example.com>
Date:   Thu Sep 11 09:27:14 2025 -0700

    Add float coercion

 config_loader/parse_config.py | 4 ++++
 tests/test_parse_config.py    | 2 +-
 2 files changed, 5 insertions(+), 1 deletion(-)

commit 6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a
Author: Marcus Lee <marcus@example.com>
Date:   Wed Sep 10 13:55:48 2025 -0700

    Support [section] headers

 config_loader/parse_config.py | 6 ++++++
 tests/test_parse_config.py    | 5 +++++
 2 files changed, 11 insertions(+)

commit 9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c
Author: Priya Raman <priya@example.com>
Date:   Tue Sep 9 16:12:09 2025 -0700

    Add make_config test helper

 tests/test_parse_config.py | 6 +++++-
 1 file changed, 5 insertions(+), 1 deletion(-)

commit c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0
Author: Marcus Lee <marcus@example.com>
Date:   Mon Sep 8 11:40:33 2025 -0700

    Parse key = value lines

 config_loader/parse_config.py | 12 ++++++++++++
 tests/test_parse_config.py    |  6 ++++++
 2 files changed, 18 insertions(+)"""

HISTORY: list[str] = [
    _FORMAT_NOTE,
    _README_DISCUSSION,
    _GREP_RESULT,
    _FIXTURE_NOTE,
    _INSTALL_LOG,
    _CI_DISCUSSION,
    _PYTEST_OUTPUT,
    _GIT_LOG,
]
