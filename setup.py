from setuptools import setup, find_packages

setup(
    name="ad_topic_recommender",
    version="0.1",
    packages=find_packages(),
    package_data={

    },
    install_requires=[
        "joblib~=1.3.2",
        "numpy~=1.24.2",
        "setuptools~=65.5.0",
        "scikit-learn~=1.3.0",
        "pandas~=2.0.3",
        "tensorflow~=2.13.0"
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "ad_topic_recommender=ad_topic_recommender:main",
        ]
    },
    author="Dilshan M. Karunarathne",
    author_email="ceo@altier.tech",
    description="A simple ad topic recommender",
    license="GNU GPLv3",
    keywords="ad topic recommender",
    url="http://www.altier.tech/"
)
