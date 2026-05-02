from setuptools import setup

setup(
    name="drp",
    version="0.1.0",
    py_modules=["drp_cli", "linter"],
    entry_points={
        "console_scripts": [
            "drp-validate=drp_cli:main",
        ]
    },
)
