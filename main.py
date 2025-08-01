import os
import io
# 'requests' is no longer needed
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime
from PIL import Image
from typing import List, Dict

# --- IMPORTS FOR AI FEATURES ---
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# --- LOAD ENVIRONMENT & CONFIGURE ---
load_dotenv()
print("GOOGLE_API_KEY (from .env):", os.getenv("GOOGLE_API_KEY"))

# Google API Key is the only AI-related key needed now
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Google API Key not found. Please set it in the .env file.")
genai.configure(api_key=API_KEY)

app = FastAPI(title="AgroSage API")

# --- CORS MIDDLEWARE ---
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- IN-MEMORY DATABASE & FACTORS (Example Data) ---
fake_db = {
    "plots": {"plot_a": {"name": "North Field", "crop": "Tomatoes", "logs": []}, "plot_b": {"name": "West Patch", "crop": "Corn", "logs": []}},
    "missions": [{"id": "m1", "title": "Start a Compost Pile", "reward": 20, "completed": False}, {"id": "m2", "title": "Apply Neem Oil", "reward": 30, "completed": False}, {"id": "m3", "title": "Install Drip Irrigation", "reward": 50, "completed": False}, {"id": "m4", "title": "Crop Rotation Plan", "reward": 25, "completed": True}],
    "carbon_ledger": [{"timestamp": "2024-03-10T10:00:00Z", "activity": "Completed Mission: Crop Rotation Plan", "credits": 25}],
    "sustainability_score": 55,
    "weather_forecast": {"condition": "High Humidity", "temperature_celsius": 28, "chance_of_rain_percent": 80}
}
EMISSION_FACTORS = {
    "tractor_diesel_per_hour": 2.7, "fertilizer_urea_per_kg": 0.8, "pesticide_per_litre": 5.0,
    "irrigation_electric_pump_per_hour": 1.5, "irrigation_diesel_pump_per_hour": 3.0, "irrigation_canal": 0.1,
    "cover_crop_reduction_factor": 0.85, "renewable_energy_reduction_factor": 0.75
}

# ==============================================================================
# --- AI MODEL INITIALIZATION ---
# ==============================================================================
print("Initializing Google Gemini AI clients...")
llm_chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY)

# CORRECTED: Switched to 'gemini-1.5-flash' to avoid the deprecation error for 'gemini-pro-vision'.
# This model handles both text and vision tasks.
llm_vision_pest = genai.GenerativeModel('gemini-1.5-flash')

# --- PROMPT ENGINEERING FIXES ---
eco_prompt_chat = ChatPromptTemplate.from_messages([("system", "You are EcoBot, a helpful assistant for Indian sustainable farming. Provide concise, actionable advice."),("human", "{input}")])
prompt_classify_waste = ChatPromptTemplate.from_template("Analyze: '{caption}'. Classify the waste type (Biodegradable, Recyclable, Electronic, etc). Respond with only the waste type.")
prompt_bin = ChatPromptTemplate.from_template("Item: '{caption}'. Based on Indian norms (SWM Rules), what dustbin color? (Green, Blue, Red, Yellow, Black). Respond with only the color.")
# This improved prompt provides more context for a better explanation
prompt_explain = ChatPromptTemplate.from_template(
    "In one simple sentence, explain why a '{item}' is considered '{category}' and goes into the {bin_color} bin in India."
)

chatbot_chain = eco_prompt_chat | llm_chat | StrOutputParser()
chain_classify_waste = prompt_classify_waste | llm_chat | StrOutputParser()
chain_bin = prompt_bin | llm_chat | StrOutputParser()
chain_explain = prompt_explain | llm_chat | StrOutputParser()
print("All AI clients initialized successfully.")

# ==============================================================================
# --- HELPER FUNCTION FOR GEMINI CAPTIONING ---
# ==============================================================================
async def get_caption_from_gemini(image_bytes: bytes) -> str:
    """Sends an image to the Google Gemini API and gets a direct caption."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # This more restrictive prompt prevents conversational fluff in the response
        prompt = (
            "Identify the main object in this image. Respond with only a short phrase describing the object, "
            "like 'a plastic water bottle' or 'a crumpled aluminum can'. "
            "Do not add any extra text, commentary, or options."
        )
        
        # This will now use the correctly initialized 'gemini-1.5-flash' model
        response = await llm_vision_pest.generate_content_async([prompt, img])
        return response.text.strip()
        
    except Exception as e:
        print(f"ERROR from Google Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"Google Gemini API Error: {str(e)}")

# ==============================================================================
# --- PYDANTIC MODELS ---
# ==============================================================================
class ChatQuery(BaseModel): query: str
class PlotLog(BaseModel): plot_id: str; soil_moisture: float; pest_sighting: str | None = None
class WasteClassificationResponse(BaseModel): caption: str; category: str; bin_color: str; explanation: str
class CarbonCalculatorInput(BaseModel):
    area: float; fertilizer_kg: float; tractor_hours: float; irrigation_hours: float;
    irrigation_type: str; cover_crop: str; renewable_energy: str
class Recommendation(BaseModel): id: str; title: str; details: str
class Alert(BaseModel): id: str; level: str; title: str; details: str
class CarbonCalculationResult(BaseModel): totalEmissions: float; categoryEmissions: Dict[str, float]; recommendations: List[Recommendation]
class DashboardDataResponse(BaseModel):
    sustainability_score: int; active_missions: int; carbon_credits: int;
    alerts: List[Alert]; recommendations: List[Recommendation]

# ==============================================================================
# --- AI RECOMMENDATION & ALERT ENGINE ---
# ==============================================================================
def generate_recommendations_and_alerts():
    recommendations, alerts = [], []
    weather = fake_db["weather_forecast"]
    if weather["chance_of_rain_percent"] > 70 and weather["temperature_celsius"] < 15:
        alerts.append(Alert(id="frost_risk_alert", level="critical", title="Frost Risk Alert", details="Low temperatures with high moisture create a high risk of frost. Protect sensitive crops immediately."))
    elif weather["condition"] == "High Humidity" and weather["temperature_celsius"] > 25:
        alerts.append(Alert(id="fungal_risk_alert", level="warning", title="High Fungal Disease Risk", details="Humid and warm conditions are ideal for fungal growth. Ensure good air circulation and monitor crops."))
    recommendations.append(Recommendation(id="rec_soil", title="Soil Health Tip", details="Consider soil testing before the next crop cycle to optimize fertilizer use and improve yield."))
    recommendations.append(Recommendation(id="rec_water", title="Water Conservation", details="Check irrigation lines for leaks to prevent water waste and reduce pumping costs."))
    return {"alerts": alerts, "recommendations": recommendations}

# ==============================================================================
# --- API ENDPOINTS ---
# ==============================================================================
@app.get("/")
def read_root(): return {"message": "AgroSage Super-App API is Live"}

@app.get("/dashboard-data", response_model=DashboardDataResponse)
def get_dashboard_data():
    generated_data = generate_recommendations_and_alerts()
    active_missions_count = len([m for m in fake_db["missions"] if not m["completed"]])
    total_credits = sum(entry['credits'] for entry in fake_db['carbon_ledger'])
    return DashboardDataResponse(
        sustainability_score=fake_db["sustainability_score"], active_missions=active_missions_count,
        carbon_credits=total_credits, alerts=generated_data["alerts"], recommendations=generated_data["recommendations"]
    )

@app.post("/calculate-carbon", response_model=CarbonCalculationResult)
async def calculate_carbon_footprint(inputs: CarbonCalculatorInput):
    category_emissions = {
        "Machinery": inputs.tractor_hours * EMISSION_FACTORS["tractor_diesel_per_hour"],
        "Fertilizers": inputs.fertilizer_kg * EMISSION_FACTORS["fertilizer_urea_per_kg"]
    }
    if inputs.irrigation_type == "Electric Pump":
        category_emissions["Irrigation"] = inputs.irrigation_hours * EMISSION_FACTORS["irrigation_electric_pump_per_hour"]
    elif inputs.irrigation_type == "Diesel Pump":
        category_emissions["Irrigation"] = inputs.irrigation_hours * EMISSION_FACTORS["irrigation_diesel_pump_per_hour"]
    else: category_emissions["Irrigation"] = inputs.irrigation_hours * EMISSION_FACTORS["irrigation_canal"]
    
    total_emissions = sum(category_emissions.values())
    if inputs.cover_crop == "Yes": total_emissions *= EMISSION_FACTORS["cover_crop_reduction_factor"]
    if inputs.renewable_energy == "Yes": total_emissions *= EMISSION_FACTORS["renewable_energy_reduction_factor"]

    recommendations = []
    if category_emissions:
        highest_source = max(category_emissions, key=category_emissions.get)
        if highest_source == "Machinery": recommendations.append(Recommendation(id="rec1", title="Machinery Tip", details="Optimize tractor routes and maintain equipment to improve fuel efficiency."))
        elif highest_source == "Fertilizers": recommendations.append(Recommendation(id="rec2", title="Fertilizer Tip", details="Use soil testing to apply precise amounts of fertilizer, avoiding overuse."))
    if total_emissions < 10: recommendations.append(Recommendation(id="rec_good", title="Great Job!", details="Your farm has low emissions! Keep up the great sustainable practices!"))
    
    return CarbonCalculationResult(totalEmissions=total_emissions, categoryEmissions=category_emissions, recommendations=recommendations)

@app.post("/classify-waste", response_model=WasteClassificationResponse)
async def classify_waste(request: Request, file: UploadFile = File(...)):
    """Takes an image, gets a caption from Google Gemini, and classifies the waste."""
    print("Headers:", request.headers)
    try:
        image_bytes = await file.read()
        caption = await get_caption_from_gemini(image_bytes)

        category = chain_classify_waste.invoke({"caption": caption}).strip()
        bin_color = chain_bin.invoke({"caption": caption}).strip()
        
        # This call now passes the full context to the improved explanation chain
        explanation = chain_explain.invoke({
            "item": caption,
            "category": category,
            "bin_color": bin_color
        }).strip()
        
        return WasteClassificationResponse(caption=caption, category=category, bin_color=bin_color, explanation=explanation)
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"An unexpected error occurred in /classify-waste: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scan-pest")
async def scan_pest(file: UploadFile = File(...)):
    """Analyzes a plant leaf image for pests or diseases."""
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        prompt = (
            "Analyze this plant leaf. "
            "1. Identify pest/disease. If healthy, say so. "
            "2. Provide a brief, organic solution. "
            "Format: 'Diagnosis: [Your Diagnosis].\\nSolution: [Your Solution].'"
        )
        # This will now use the correctly initialized 'gemini-1.5-flash' model
        response = await llm_vision_pest.generate_content_async([prompt, image])
        return {"result": response.text.strip()}
    except Exception as e:
        print(f"Error in /scan-pest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/missions")
def get_missions(): return fake_db["missions"]

@app.post("/ask-ecobot")
async def ask_bot(request: ChatQuery):
    """Provides answers to farming questions via the EcoBot LLM chain."""
    try:
        answer = chatbot_chain.invoke({"input": request.query})
        return {"response": answer}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/complete-mission/{mission_id}")
def complete_mission(mission_id: str):
    """Marks a mission as complete and updates the user's score and ledger."""
    mission = next((m for m in fake_db["missions"] if m["id"] == mission_id), None)
    if not mission or mission["completed"]: raise HTTPException(status_code=404, detail="Mission not found or already completed")
    mission["completed"] = True
    entry = {"timestamp": datetime.now().isoformat(), "activity": f"Completed Mission: {mission['title']}", "credits": mission['reward']}
    fake_db["carbon_ledger"].append(entry)
    fake_db["sustainability_score"] += mission['reward']
    return {"message": "Mission completed!", "entry": entry}

@app.post("/log-plot-data")
async def log_plot_data(log: PlotLog):
    """Logs new data for a specific farm plot."""
    if log.plot_id not in fake_db["plots"]: raise HTTPException(status_code=404, detail="Plot not found")
    timestamped_log = log.model_dump()
    timestamped_log["timestamp"] = datetime.now().isoformat()
    fake_db["plots"][log.plot_id]["logs"].append(timestamped_log)
    return {"message": "Log received successfully."}

# The uvicorn runner is for local development only. Render will use its own command.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)