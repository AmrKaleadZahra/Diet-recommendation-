from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import os

# === Paths ===
DATA_FILE = "recipes_with_prices21.csv.gz"
SCALER_FILE = "scaler3.pkl"
MODEL_FILE = "diet_model00.keras"

# === Global variables ===
model = None
recipes_df = None
scaler = None
encoded_recipes = None
resources_loaded = False

# === FastAPI App ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    global model, recipes_df, scaler, encoded_recipes, resources_loaded

    print("🚀 Starting resource loading...")

    try:
        print("📦 Loading model...")
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found.")
        model = tf.keras.models.load_model(MODEL_FILE, compile=False)

        print("📄 Loading recipes data...")
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"Data file '{DATA_FILE}' not found.")
        selected_columns = ['Calories', 'Keywords', 'Name', 'MealType', 'EstimatedPriceEGP',
                            'FatContent', 'SaturatedFatContent', 'CholesterolContent',
                            'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent',
                            'ProteinContent', 'RecipeIngredientQuantities', 'RecipeIngredientParts']
        recipes_df = pd.read_csv(DATA_FILE, usecols=selected_columns, compression='gzip')

        print("⚙️ Loading scaler...")
        if not os.path.exists(SCALER_FILE):
            raise FileNotFoundError(f"Scaler file '{SCALER_FILE}' not found.")
        scaler = joblib.load(SCALER_FILE)

        if os.path.exists('encoded_recipes.npy'):
            print("📥 Loading encoded recipes from file...")
            encoded_recipes = np.load('encoded_recipes.npy')
        else:
            print("🔢 Encoding recipes (first time)...")
            nutrition_columns = [
                'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
                'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent'
            ]
            scaled_data = scaler.transform(recipes_df[nutrition_columns])
            encoded_recipes = model.predict(scaled_data)
            np.save('encoded_recipes.npy', encoded_recipes)

        resources_loaded = True
        print("✅ Resources loaded successfully!")

    except Exception as e:
        print(f"❌ Error during startup: {e}")
        resources_loaded = False



# === Request Model ===
class UserInput(BaseModel):
    gender: str
    weight: float
    height: float
    age: int
    activity_level: str
    goal: str
    daily_budget: float
    dietary_restrictions: list[str]

# === Helper Functions ===
def compute_bmr(gender, weight, height, age):
    if gender == 'male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    elif gender == 'female':
        return 10 * weight + 6.25 * height - 5 * age - 161
    else:
        raise ValueError("Invalid gender")

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
    return round(bmr * intensity_multipliers.get(activity_level, 1.2) * objective_adjustments.get(goal, 1))

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

    return similar_recipes[['Name', 'MealType', 'Calories', 'EstimatedPriceEGP', 'RecipeIngredientParts', 'RecipeIngredientQuantities']].head(top_n)

def suggest_full_day_meal_plan(total_calories, daily_budget, dietary_restrictions=None, top_n=5):
    meal_types = ['breakfast', 'snack', 'lunch', 'dinner']
    plan = {}
    for meal in meal_types:
        recipes = suggest_recipes(total_calories, meal, daily_budget, dietary_restrictions, top_n)
        if not recipes.empty:
            plan[meal] = recipes.head(1).reset_index(drop=True)
        else:
            plan[meal] = pd.DataFrame([{
                'Name': 'No meal found',
                'MealType': meal,
                'Calories': None,
                'EstimatedPriceEGP': None,
                'RecipeIngredientParts': None,
                'RecipeIngredientQuantities': None
            }])
    return plan

# === Routes ===
@app.post("/personalized_recommend")
def personalized_recommendation(user: UserInput):
    if not resources_loaded:
        raise HTTPException(status_code=503, detail="Resources are still loading, try again later.")
    
    bmr = compute_bmr(user.gender, user.weight, user.height, user.age)
    target_calories = compute_daily_caloric_intake(bmr, user.activity_level, user.goal)

    suggestions = suggest_full_day_meal_plan(
        total_calories=target_calories,
        daily_budget=user.daily_budget,
        dietary_restrictions=user.dietary_restrictions
    )

    return {
        "daily_calories": target_calories,
        "per_meal_target": round(target_calories / 5),
        "suggested_recipes": {meal: df.to_dict(orient="records") for meal, df in suggestions.items()}
    }

@app.get("/")
def read_root():
    return {"message": "Service is running 🚀"}
