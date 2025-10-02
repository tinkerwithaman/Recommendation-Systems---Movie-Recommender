# Movie Recommender System

## Description
This project implements a collaborative filtering-based movie recommender system using the `scikit-surprise` library. It uses the famous **Singular Value Decomposition (SVD)** algorithm to predict movie ratings a user might give and then generates top-N recommendations.

The script automatically downloads and uses the MovieLens 100k dataset, which contains 100,000 ratings from 943 users on 1682 movies.

## Features
-   Uses the powerful `scikit-surprise` library for building recommenders.
-   Implements the SVD algorithm for matrix factorization.
-   Automatically downloads and caches the MovieLens 100k dataset.
-   Evaluates the model using Root Mean Squared Error (RMSE).
-   Generates top 5 movie recommendations for a specified user.

## Setup and Installation

1.  **Clone the repository and navigate to the directory.**
2.  **Create a virtual environment and activate it.**
3.  **Install the dependencies:** `pip install -r requirements.txt`
4.  **Run the script:** `python src/main.py`
    *(Note: The first run will download the MovieLens 100k dataset, which may take a moment.)*

## Example Output
```
Loading MovieLens 100k dataset...
Evaluating SVD model...
RMSE: 0.9357
---
Top 5 recommendations for user 196:
1. Movie ID 408: 'Close Shave, A (1995)' with predicted rating 5.00
2. Movie ID 318: 'Schindler's List (1993)' with predicted rating 4.95
3. Movie ID 169: 'Wrong Trousers, The (1993)' with predicted rating 4.89
4. Movie ID 483: 'Casablanca (1942)' with predicted rating 4.87
5. Movie ID 114: 'Wallace & Gromit: The Best of Aardman Animation (1996)' with predicted rating 4.82
```
