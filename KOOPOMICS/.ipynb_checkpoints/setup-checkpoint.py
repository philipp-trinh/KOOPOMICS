from setuptools import setup, find_packages


install_requires=[
    "torch==2.2.2",
    "matplotlib==3.9.2",
    "numpy==1.26.4",
    "scikit-learn==1.5.2",
    "umap-learn==0.5.6",
    "pandas==2.2.3",
    "plotly==5.24.1",
    "seaborn==0.13.2",
    "torchvision==0.17.2"
]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="KOOPOMICS",
    version="1.0.0",
    description="Koopman models for OMICS timeseries analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.10, <3.11",
    zip_safe=False,
)
