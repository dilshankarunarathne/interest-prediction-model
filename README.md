# Ad Topic Recommendation System [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]
 An AI model for suggesting interested topics for social media advertising, based on the user's age and gender.

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].
[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Model](#model)
5. [Results](#results)
6. [Conclusion](#conclusion)
7. [References](#references)
8. [License](#license)
9. [Author Info](#author-info)
10. [Contributors](#contributors)

## Introduction

This document provides an overview of an AI model designed to recommend topics of interest based on a 
person's age and gender. The model utilizes a Decision Tree Classifier to make topic recommendations. 
The following sections detail the dataset used, data preprocessing, model development, results, and conclusions.

## Dataset

The dataset used for training and evaluating the recommendation model is stored in 'generated_dataset.csv.' It contains the following columns:

- `UserAge`: Represents the age of the users.
- `UserGender`: Categorizes users by gender (encoded numerically).
- `LikedTopic`: Indicates the topic of interest that users have expressed.
