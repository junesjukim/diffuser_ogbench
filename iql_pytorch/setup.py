from setuptools import setup, find_packages

setup(
    name="iql_pytorch",
    version="0.1",
    packages=find_packages(include=['iql_pytorch', 'iql_pytorch.*']),
    install_requires=[
        "torch",
        "numpy",
    ],
) 