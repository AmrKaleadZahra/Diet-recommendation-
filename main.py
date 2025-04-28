from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import threading
import os

# === مسارات الملفات ===
DATA_FILE = "recipes_with_prices21.csv.gz"
SCALER_FILE = "scaler3.pkl"
MODEL_FILE = "diet_model00.keras"

# === متغيرات عامة ===
model = None
recipes_df = None
scaler = None
encoded_recipes = None
resources_loaded = False

# === FastAPI app ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Event: Startup ===
@app.on_event("startup")
def startup_event():
    def load_resources():
        global model, recipes_df, scaler, encoded_recipes, resources_loaded
        print("📦 Loading model...")
        model = tf.keras.models.load_model(MODEL_FILE, compile=False)

        print("📄 Loading recipes data...")
        selected_columns = ['Calories', 'Keywords', 'Name', 'MealType', 'EstimatedPriceEGP',
                            'FatContent', 'SaturatedFatContent', 'CholesterolContent',
                            'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent',
                            'ProteinContent', 'RecipeIngredientQuantities', 'RecipeIngredientParts']
        recipes_df = pd.read_csv(DATA_FILE, usecols=selected_columns, compression='gzip')

        print("⚙️ Loading scaler...")
        scaler = joblib.load(SCALER_FILE)

        nutrition_columns = [
            'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
            'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent'
        ]
        scaled_data = scaler.transform(recipes_df[nutrition_columns])
        encoded_recipes = model.predict(scaled_data)

        resources_loaded = True
        print("✅ Resources loaded successfully!")

    threading.Thread(target=load_resources).start()

# === Request model ===
class UserInput(BaseModel):
    gender: str
    weight: float
    height: float
    age: int
    activity_level: str
    goal: str
    daily_budget: float
    dietary_restrictions: list[str]

# === BMR & Calories Calculation ===
def compute_bmr(gender, weight, height, age):
    if gender == 'male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    elif gender == 'female':
        return 10 * weight + 6.25 * height - 5 * age - 161
    else:
        raise ValueError("Invalid gender: should be 'male' or 'female'.")

def compute_daily_caloric_intake(bmr, activity_level, goal):
    intensity_multipliers = {
        'sedentary': 1.2,
        'lightly_active': 1.375,
        'moderately_active': 1.55,
        'very_active': 1.725,
        'extra_active': 1.9
    }
    objective_adjustments = {
        'weight_loss': 0.8,
        'muscle_gain': 1.2,
        'health_maintenance': 1
    }
    if activity_level not in intensity_multipliers:
        raise ValueError(f"Invalid activity_level: {activity_level}")
    if goal not in objective_adjustments:
        raise ValueError(f"Invalid goal: {goal}")

    maintenance_calories = bmr * intensity_multipliers[activity_level]
    total_caloric_intake = maintenance_calories * objective_adjustments[goal]
    return round(total_caloric_intake)

# === Recipe Suggestion ===
def suggest_recipes(total_calories, meal_type, daily_budget, dietary_restrictions, top_n=5):
    meal_split = {
        'breakfast': (0.20, 0.20),
        'snack':     (0.15, 0.15),
        'lunch':     (0.35, 0.35),
        'dinner':    (0.30, 0.30)
    }
    cal_ratio, budget_ratio = meal_split.get(meal_type.lower(), (0.25, 0.25))
    target_calories = total_calories * cal_ratio
    target_budget = daily_budget * budget_ratio

    user_input_features = np.array([[target_calories, 0, 0, 0, 0, 0, 0, 0, 0]])
    scaled_input_features = scaler.transform(user_input_features)
    predicted_latent_features = model.predict(scaled_input_features)

    similarities = cosine_similarity(predicted_latent_features, encoded_recipes)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    similar_recipes = recipes_df.iloc[top_indices].copy()

    if 'MealType' in similar_recipes.columns:
        similar_recipes = similar_recipes[similar_recipes['MealType'].str.lower() == meal_type.lower()]

    similar_recipes = similar_recipes[
        (similar_recipes['EstimatedPriceEGP'] <= target_budget) &
        (similar_recipes['Calories'] <= target_calories)
    ]

    similar_recipes = similar_recipes.sort_values(by=['EstimatedPriceEGP', 'Calories'], ascending=[False, False])

    if dietary_restrictions:
        pattern = '|'.join([r.lower() for r in dietary_restrictions])
        similar_recipes = similar_recipes[
            ~similar_recipes['Name'].str.lower().str.contains(pattern, na=False)
        ]
        if 'RecipeIngredientParts' in similar_recipes.columns:
            similar_recipes = similar_recipes[
                ~similar_recipes['RecipeIngredientParts'].str.lower().str.contains(pattern, na=False)
            ]
        if 'Keywords' in similar_recipes.columns:
            similar_recipes = similar_recipes[
                ~similar_recipes['Keywords'].str.lower().str.contains(pattern, na=False)
            ]

    if similar_recipes.empty:
        fallback = recipes_df.copy()
        fallback = fallback[fallback['MealType'].str.lower() == meal_type.lower()]
        fallback = fallback[fallback['EstimatedPriceEGP'] <= target_budget]
        fallback['CalorieDiff'] = np.abs(fallback['Calories'] - target_calories)

        if dietary_restrictions:
            pattern = '|'.join([r.lower() for r in dietary_restrictions])
            fallback = fallback[~fallback['Name'].str.lower().str.contains(pattern, na=False)]
            if 'RecipeIngredientParts' in fallback.columns:
                fallback = fallback[
                    ~fallback['RecipeIngredientParts'].astype(str).str.lower().str.contains(pattern, na=False)
                ]
            if 'Keywords' in fallback.columns:
                fallback = fallback[
                    ~fallback['Keywords'].astype(str).str.lower().str.contains(pattern, na=False)
                ]
        
        return fallback.sort_values(by='CalorieDiff').head(top_n)[['Name', 'MealType', 'Calories', 'EstimatedPriceEGP', 'RecipeIngredientParts','RecipeIngredientQuantities']]

    return similar_recipes[['Name', 'MealType', 'Calories', 'EstimatedPriceEGP', 'RecipeIngredientParts','RecipeIngredientQuantities']].head(top_n)

def suggest_full_day_meal_plan(total_calories, daily_budget, dietary_restrictions=None, top_n=5):
    meal_types = ['breakfast', 'snack', 'lunch', 'dinner']
    plan = {}

    for meal in meal_types:
        meal_recommendation = suggest_recipes(
            total_calories=total_calories,
            meal_type=meal,
            daily_budget=daily_budget,
            dietary_restrictions=dietary_restrictions,
            top_n=top_n
        )

        if not meal_recommendation.empty:
            plan[meal] = meal_recommendation.head(1).reset_index(drop=True)
        else:
            plan[meal] = pd.DataFrame([{
                'Name': 'No suitable meal found',
                'MealType': meal,
                'Calories': None,
                'EstimatedPriceEGP': None,
                'RecipeIngredientParts': None,
                'RecipeIngredientQuantities':None
            }])
    
    return plan

# === FastAPI Route ===
@app.post("/personalized_recommend")
def personalized_recommendation(user: UserInput):
    bmr = compute_bmr(user.gender, user.weight, user.height, user.age)
    target_calories = compute_daily_caloric_intake(bmr, user.activity_level, user.goal)
    per_meal_calories = target_calories / 5

    suggestions = suggest_full_day_meal_plan(
        total_calories=target_calories,
        daily_budget=user.daily_budget,
        dietary_restrictions=user.dietary_restrictions
    )

    return {
        "daily_calories": round(target_calories),
        "per_meal_target": round(per_meal_calories),
        "suggested_recipes": {
            meal: df.to_dict(orient='records') for meal, df in suggestions.items()
        }
    }

@app.get("/")
def read_root():
    return {"message": "Service is running 🚀"}

