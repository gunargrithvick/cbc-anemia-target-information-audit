"""Run the fast, reproducible project-quality checks.

This command deliberately does not rerun the long research pipeline.
Use ``python src/run_all.py`` when source or dependency inputs change; this
check then verifies the resulting artefacts and exercises the safety rules.
"""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run(label, args):
    """Run one check from the project root and return its exit status."""
    print(f"\n=== {label} ===")
    result = subprocess.run(args, cwd=ROOT)
    if result.returncode:
        print(f"{label} failed with exit code {result.returncode}")
    else:
        print(f"{label} passed")
    return result.returncode


def main():
    checks = [
        ("tests and coverage", [
            sys.executable, "-m", "pytest", "tests", "-q", "-W", "error",
            "--cov=src", "--cov-report=term-missing",
            "--cov-report=json:coverage.json",
        ]),
        ("static name checks", [
            sys.executable, "-m", "pyflakes", "src", "tests",
        ]),
        ("dependency consistency", [
            sys.executable, "-m", "pip", "check",
        ]),
    ]
    failures = [run(label, args) for label, args in checks]
    if any(failures):
        return 1
    print("\nAll quality checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
