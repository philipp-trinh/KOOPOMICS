from setuptools import setup, find_packages

# Runtime dependencies
install_requires = [
    "numpy==1.26.4",
    "torch==2.0.0",    
    "torchvision==0.15.1",
    "matplotlib==3.9.2",
    "scikit-learn==1.5.2",
    "umap-learn==0.5.6",
    "pandas==2.2.3",
    "plotly==5.24.1",
    "seaborn==0.13.2",
    "wandb==0.18.5",
    "numba>=0.56.0",
    "ipython>=8.0.0",
    "ipywidgets>=8.0.0",
    "jupyter>=1.0.0",
    "captum>=0.6.0",
    "torchviz>=0.0.2"
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
