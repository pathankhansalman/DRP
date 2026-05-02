from setuptools import find_packages, setup

setup(
    name="drp-protocol",
    version="0.1.0",
    description="Decision Record Protocol CLI and lint tooling",
    packages=find_packages(),
    py_modules=["drp_cli", "linter"],
    entry_points={"console_scripts": ["drp-validate=drp_cli:main"]},
    python_requires=">=3.10",
)
