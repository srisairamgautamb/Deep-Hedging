from setuptools import setup, find_packages

setup(
    name="deep-hedging",
    version="1.0.0",
    description="Deep RL hedging for exotic barrier options",
    author="Quantitative Research",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.4.0",
        "scipy>=1.8.0",
        "torch>=2.0.0",
        "gymnasium>=0.28.0",
        "pyarrow>=10.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
    },
)
