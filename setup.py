from setuptools import setup, find_packages

setup(
    name="geoguessr-classifier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "kagglehub",
        "numpy",
        "matplotlib",
        "tqdm",
        "Pillow",
        "scikit-learn",
        "torchinfo",
    ],
    python_requires=">=3.8",
)
