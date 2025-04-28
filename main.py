from fastapi import FastAPI, HTTPException
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
import threading
import time

# === إعداد Google Drive ===
file_id = '1Asnxs5veFwsLcAnmquHgFbRukfC5DGTD'
url = f"https://drive.google.com/uc?id={file_id}"
local_filename = "recipes_with_prices2.csv.gz"

# موارد عالمية
model = None
recipes_df = None
scaler = None
encoded_recipes = None
resources_loaded = False

# تحميل الموارد
def load_resources():
    global model, recipes_df, scaler, encoded_recipes, resources_loaded

    try:
        # تحميل الملف لو مش موجود
        if not os.path.exists(local_filename):
            gdown.download(url, local_filename, quiet=False)

        # تحميل البيانات
        selected_columns = ['Calories', 'Keywords', 'Name', 'MealType', 'EstimatedPriceEGP',
                            'FatContent', 'SaturatedFatContent', 'CholesterolContent',
                            'SodiumContent', 'CarbohydrateContent', 'FiberContent',
                            'SugarContent', 'ProteinContent', 'RecipeIngredientQuantities',
                            'RecipeIngredientParts']
        recipes_df = pd.read_csv(local_filename, usecols=selected_columns, compression='gzip')

        # تحميل النموذج
        model = tf.keras.models.load_model("diet_model00.keras", compile=False)

        # تحميل الـScaler
        scaler = joblib.load("scaler3.pkl")

        # تجهيز الـ encoded recipes
        nutrition_columns = ['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
                             'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent',
                             'ProteinContent']
        scaled_data = scaler.transform(recipes_df[nutrition_columns])
        encoded_recipes = model.predict(scaled_data)

        resources_loaded = True
    except Exception as e:
        print(f"Error loading resources: {e}")
        resources_loaded = False

# تشغيل التحميل في الخلفية
loading_thread = threading.Thread(target=load_resources)
loading_thread.start()

# === إعداد FastAPI ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === الموديل ===
class UserInput(BaseModel):
    gender: str
    weight: float
    height: float
    age: int
    activity_level: str
    goal: str
    daily_budget: float
    dietary_restrictions: list[str]

# === دوال مساعدة ===
def compute_bmr(gender, body_weight, body_height, age):
    if gender == 'male':
        return 10 * body_weight + 6.25 * body_height - 5 * age + 5
    elif gender == 'female':
        return 10 * body_weight + 6.25 * body_height - 5 * age - 161
    else:
        raise ValueError("Invalid gender. Please choose 'male' or 'female'.")

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

    maintenance = bmr * intensity_multipliers[activity_level]
    return round(maintenance * objective_adjustments[goal])

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

    user_features = np.array([[target_calories, 0, 0, 0, 0, 0, 0, 0, 0]])
    scaled_features = scaler.transform(user_features)
    predicted_latent = model.predict(scaled_features)

    similarities = cosine_similarity(predicted_latent, encoded_recipes)[0]
    top_indices = np.argsort(similarities)[::-1]

    candidates = recipes_df.iloc[top_indices].copy()

    candidates = candidates[
        (candidates['MealType'].str.lower() == meal_type.lower()) &
        (candidates['EstimatedPriceEGP'] <= target_budget) &
        (candidates['Calories'] <= target_calories)
    ]

    if dietary_restrictions:
        pattern = '|'.join([r.lower() for r in dietary_restrictions])
        candidates = candidates[~candidates['Name'].str.lower().str.contains(pattern, na=False)]

    if candidates.empty:
        fallback = recipes_df.copy()
        fallback = fallback[fallback['MealType'].str.lower() == meal_type.lower()]
        fallback = fallback[fallback['EstimatedPriceEGP'] <= target_budget]
        fallback['CalorieDiff'] = np.abs(fallback['Calories'] - target_calories)

        if dietary_restrictions:
            pattern = '|'.join([r.lower() for r in dietary_restrictions])
            fallback = fallback[~fallback['Name'].str.lower().str.contains(pattern, na=False)]

        return fallback.sort_values('CalorieDiff').head(top_n)

    return candidates.head(top_n)

def suggest_full_day_meal_plan(total_calories, daily_budget, dietary_restrictions=None, top_n=5):
    meals = ['breakfast', 'snack', 'lunch', 'dinner']
    plan = {}
    for meal in meals:
        recipes = suggest_recipes(
            total_calories=total_calories,
            meal_type=meal,
            daily_budget=daily_budget,
            dietary_restrictions=dietary_restrictions or [],
            top_n=top_n
        )
        if recipes.empty:
            plan[meal] = pd.DataFrame([{
                'Name': 'No suitable meal found',
                'MealType': meal,
                'Calories': None,
                'EstimatedPriceEGP': None,
                'RecipeIngredientParts': None,
                'RecipeIngredientQuantities': None
            }])
        else:
            plan[meal] = recipes.reset_index(drop=True).iloc[[0]]
    return plan

# === الراوت ===
@app.post("/personalized_recommend")
async def personalized_recommendation(user: UserInput):
    if not resources_loaded:
        raise HTTPException(status_code=503, detail="Resources are still loading, please try again later.")

    bmr = compute_bmr(user.gender, user.weight, user.height, user.age)
    target_calories = compute_daily_caloric_intake(bmr, user.activity_level, user.goal)
    per_meal_calories = target_calories / 5

    meal_plan = suggest_full_day_meal_plan(
        total_calories=target_calories,
        daily_budget=user.daily_budget,
        dietary_restrictions=user.dietary_restrictions
    )

    return {
        "daily_calories": round(target_calories),
        "per_meal_target": round(per_meal_calories),
        "suggested_recipes": {
            meal: df.to_dict(orient='records') for meal, df in meal_plan.items()
        }
    }
