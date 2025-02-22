from fastapi import FastAPI
import pandas as pd
from model import recommend_movies_weighted
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/recommendation/{prompt}")
async def recommendation(prompt):
    column_weights = {
        'overview': 2.0,
        'genres': 3.0,
        'keywords': 2.5,
        'tagline': 0.0,
        'cast': 0.0,
        'director': 0.0,
        'title_list': 2.0
    }
    movies_df = pd.read_csv('movies_cleaned_preprocessed.csv')
    recommendations = recommend_movies_weighted(prompt, movies_df, column_weights, top_n=5, reg=True)
    recommendations = transform_recommendations(recommendations)
    return recommendations


def transform_recommendations(df):
    return [
        {
            "title": row['title'],
            "genres": row['genres'],
            "cast": row['cast'],
            "similarity_score": float(row['similarity_score'])
        }
        for _, row in df.iterrows()
    ]




