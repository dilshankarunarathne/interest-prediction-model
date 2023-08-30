# Ad Topic Recommendation System 
 An AI model for suggesting interested topics for social media advertising, based on the user's age and gender.

[![Version](https://img.shields.io/badge/version-0.1-brightgreen.svg)](https://pypi.org/project/ad-topic-recommender/)
[![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Python Library](#python-library)
- [Example](#example)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Contact Information](#contact-information)

## Introduction

The `ad_topic_recommender` package is a simple tool that recommends interesting topics based on a person's age and gender. This documentation provides information on how to install and use the package, along with examples and other relevant details.

## Installation

You can install the `ad_topic_recommender` package using pip:

```bash
pip install ad-topic-recommender
```

## Usage

### Command Line Interface

The package provides a command-line interface (CLI) for easy topic recommendations. You can use it as follows:

```bash
ad_topic_recommender --age <age> --gender <gender>
```

Replace <age> with the person's age and <gender> with their gender (e.g., "male" or "female"). 
The CLI will then print out a recommended topic.

### Python Library

You can also use the package as a Python library in your own projects. Here's an example of how to use it:

```python
from ad_topic_recommender import recommend_topic

age = 30
gender = "male"

topic = recommend_topic(age, gender)
print(f"Recommended topic for a {age}-year-old {gender}: {topic}")
```

## Example

Here's an example of how to use the ad_topic_recommender package in a Python script:

```python
from ad_topic_recommender import recommend_topic

age = 25
gender = "female"

topic = recommend_topic(age, gender)
print(f"Recommended topic for a {age}-year-old {gender}: {topic}")
```

## Dependencies

The package has the following dependencies:

- joblib~=1.3.2
- numpy~=1.24.2
- setuptools~=65.5.0
- scikit-learn~=1.3.0
- pandas~=2.0.3
- tensorflow~=2.13.0

Make sure these dependencies are installed in your environment.

## Contributing

If you'd like to contribute to this project, please check the contribution guidelines for more information.

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]  
[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa] 

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

## Contact Information

For questions or feedback, please contact the author:

- Author: Dilshan M. Karunarathne
- Email: ceo@altier.tech
- Website: [http://altier.tech](http://altier.tech)
