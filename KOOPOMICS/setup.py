from setuptools import setup, find_packages

install_requires = [

]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="KOOPOMICS",
    version="1.0.0",
    description="Koopman models of OMICS data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.9",
    zip_safe=False,
)
