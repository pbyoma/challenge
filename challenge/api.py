import fastapi
from challenge.model import DelayModel

app = fastapi.FastAPI()
model = DelayModel()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict() -> dict:
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame(data)
        
        # Preprocess the data
        features = model.preprocess(df)
        
        # Make predictions
        predictions = model.predict(features)
        
        # Return predictions as a response
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))