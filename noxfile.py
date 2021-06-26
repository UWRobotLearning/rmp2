"""Nox sessions."""
# from pathlib import Path

import nox
from nox.sessions import Session

project = "robolearn"
package = "robolearn"
package_path = f"{project}/{package}"
tests_path = f"{project}/tests"

# NOTE: Dev Requirements are defined in setup.py
# however we explicitly list just those needed for tests here as our
# workflow only install reqs needed for a specific session, not all dev reqs.
test_reqs = ["coverage", "pytest", "pygments"]

python_versions = ["3.9", "3.8", "3.7"]

nox.options.sessions = (
    "pre-commit",
    "tests",
)


@nox.session(name="pre-commit", python="3.9")
def precommit(session: Session) -> None:
    """Lint using pre-commit."""
    session.install(
        "black",
        "flake8",
        "flake8-bugbear",
        "pep8-naming",
        "pre-commit",
        "pre-commit-hooks",
        "reorder-python-imports",
    )
    session.run("pre-commit", "run", "--all-files", "--show-diff-on-failure")


@nox.session(python=python_versions)
def tests(session: Session) -> None:
    """Run the test suite."""
    session.install(".")
    session.install(*test_reqs)
    session.run("pip", "install", ".[dev]")

    try:
        session.run("coverage", "run", "-m", "pytest", *session.posargs)
    finally:
        session.notify("coverage_report")


@nox.session
def coverage_report(session: Session) -> None:
    """Generate coverage report from tests output."""
    session.install("coverage[toml]")

    if any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")

    session.run("coverage", "report")  # show coverage report in CLI
    session.run("coverage", "xml")  # save report to xml for upload to codecov
