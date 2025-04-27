import time
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import gdown
import os

# === إعداد Google Drive ===
file_id = '1WW1XU9TTXE8BWb0Nr1IcfyMOqUlzuj5B'
url = f"https://drive.google.com/uc?id={file_id}"
local_filename = "recipes_with_prices5.csv"

if not os.path.exists(local_filename):
    gdown.download(url, local_filename, quiet=False)

# === تحميل الموديل والسكيلر ===
model = tf.keras.models.load_model("diet_model00.keras", compile=False)
scaler = joblib.load("scaler3.pkl")

nutrition_columns = [
    'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
    'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent'
]

# === FastAPI App ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    gender: str
    weight: float
    height: float
    age: int
    activity_level: str
    goal: str
    daily_budget: float
    dietary_restrictions: list[str]

# === الكاش مع توقيت التخزين
meal_cache = {}
CACHE_EXPIRATION_SECONDS = 300  # 5 دقائق

def set_cache(key, value):
    meal_cache[key] = {
        'timestamp': time.time(),
        'value': value
    }

def get_cache(key):
    item = meal_cache.get(key)
    if not item:
        return None
    # تحقق من مدة الصلاحية
    if time.time() - item['timestamp'] > CACHE_EXPIRATION_SECONDS:
        del meal_cache[key]
        return None
    return item['value']

# === حسابات السعرات
def compute_bmr(gender, body_weight, body_height, age):
    if gender == 'male':
        return 10 * body_weight + 6.25 * body_height - 5 * age + 5
    elif gender == 'female':
        return 10 * body_weight + 6.25 * body_height - 5 * age - 161
    else:
        raise ValueError("Invalid gender.")

def compute_daily_caloric_intake(bmr, activity_intensity, objective):
    intensity_multipliers = {
        'sedentary': 1.2, 'lightly_active': 1.375,
        'moderately_active': 1.55, 'very_active': 1.725, 'extra_active': 1.9
    }
    objective_adjustments = {
        'weight_loss': 0.8, 'muscle_gain': 1.2, 'health_maintenance': 1
    }
    return round(bmr * intensity_multipliers[activity_intensity] * objective_adjustments[objective])

# === تحميل بيانات على الطلب
def load_recipes_batch(meal_type=None):
    df = pd.read_csv(local_filename, usecols=[
        'Name', 'MealType', 'Calories', 'EstimatedPriceEGP', 'RecipeIngredientParts'
    ] + nutrition_columns)

    if meal_type:
        df = df[df['MealType'].str.lower() == meal_type.lower()]
    return df

def suggest_recipes(total_calories, meal_type, daily_budget, dietary_restrictions, top_n=5):
    cache_key = f"{meal_type}_{total_calories}_{daily_budget}_{'_'.join(dietary_restrictions)}"
    cached = get_cache(cache_key)
    if cached is not None:
        return cached

    recipes_df = load_recipes_batch(meal_type)
    if recipes_df.empty:
        return pd.DataFrame()

    input_features = recipes_df[nutrition_columns]
    scaled_data = scaler.transform(input_features)
    encoded_recipes = model.predict(scaled_data)

    meal_split = {
        'breakfast': (0.20, 0.20),
        'snack':     (0.15, 0.15),
        'lunch':     (0.35, 0.35),
        'dinner':    (0.30, 0.30)
    }
    cal_ratio, budget_ratio = meal_split.get(meal_type.lower(), (0.25, 0.25))
    target_calories = total_calories * cal_ratio
    target_budget = daily_budget * budget_ratio

    user_input_features = np.array([[target_calories] + [0]*8])
    scaled_input_features = scaler.transform(user_input_features)
    predicted_latent_features = model.predict(scaled_input_features)

    similarities = cosine_similarity(predicted_latent_features, encoded_recipes)[0]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    similar_recipes = recipes_df.iloc[top_indices].copy()

    similar_recipes = similar_recipes[
        (similar_recipes['EstimatedPriceEGP'] <= target_budget) &
        (similar_recipes['Calories'] <= target_calories)
    ]

    if dietary_restrictions:
        pattern = '|'.join([r.lower() for r in dietary_restrictions])
        similar_recipes = similar_recipes[~similar_recipes['Name'].str.lower().str.contains(pattern, na=False)]

    result = similar_recipes[['Name', 'MealType', 'Calories', 'EstimatedPriceEGP', 'RecipeIngredientParts']].head(top_n)

    # تخزين النتيجة في الكاش مع توقيت
    set_cache(cache_key, result)
    return result

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
                'RecipeIngredientParts': None
            }])

    return plan

# === FastAPI Route
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
