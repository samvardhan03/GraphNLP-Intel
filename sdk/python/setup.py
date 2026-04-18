from setuptools import setup, find_packages
setup(
    name="graphnlp-client",
    version="0.1.3",
    packages=find_packages(),
    long_description=open("../../README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=["httpx>=0.27"],
)
