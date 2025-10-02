import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split

def get_movie_titles(movie_id_list, movies_df):
    """Fetches movie titles from their IDs."""
    titles = []
    for movie_id in movie_id_list:
        title = movies_df[movies_df['item_id'] == int(movie_id)]['title'].values[0]
        titles.append(title)
    return titles

def main():
    # --- 1. Load Data ---
    print("Loading MovieLens 100k dataset...")
    # The 'surprise' library can download and load the dataset automatically.
    data = Dataset.load_builtin('ml-100k')

    # To get movie names, we need to load the u.item file manually.
    # The file path can be found in the loaded dataset object.
    movies_filepath = f"{data.path}/u.item"
    movies_df = pd.read_csv(
        movies_filepath, 
        sep='|', 
        encoding='latin-1', 
        header=None,
        names=['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + [f'genre_{i}' for i in range(19)]
    )

    # --- 2. Train and Evaluate the Model ---
    print("Evaluating SVD model...")
    # Use the SVD algorithm (a form of matrix factorization)
    algo = SVD()
    
    # Run 5-fold cross-validation and print the results
    # This evaluates the model's prediction accuracy (RMSE)
    cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
    mean_rmse = cv_results['test_rmse'].mean()
    print(f"RMSE: {mean_rmse:.4f}")

    # For making predictions, we train on the full dataset
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    # --- 3. Generate Recommendations for a User ---
    # Let's predict ratings for a specific user
    user_id_to_recommend = '196' # This is a string user ID from the dataset
    
    # Get a list of all movie IDs
    all_movie_ids = trainset.all_items()
    
    # Get a list of movies the user has already rated
    movies_rated_by_user = [item for (item, rating) in trainset.ur[trainset.to_inner_uid(user_id_to_recommend)]]

    # Predict ratings for all movies the user hasn't seen
    predictions = []
    for movie_id in all_movie_ids:
        if movie_id not in movies_rated_by_user:
            predicted_rating = algo.predict(user_id_to_recommend, trainset.to_raw_iid(movie_id)).est
            predictions.append((trainset.to_raw_iid(movie_id), predicted_rating))

    # Sort the predictions by rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get the top N recommendations
    top_n = 5
    top_recommendations = predictions[:top_n]
    
    print(f"\n---\nTop {top_n} recommendations for user {user_id_to_recommend}:")
    for movie_id, rating in top_recommendations:
        movie_title = movies_df[movies_df['item_id'] == int(movie_id)]['title'].values[0]
        print(f"{rating:.2f} - {movie_title} (ID: {movie_id})")
        
if __name__ == "__main__":
    main()
