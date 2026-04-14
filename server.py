import mlx.core as mx
from mlx_lm import load, generate
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- API Configuration ---
app = FastAPI(title="Shrine Seeker Compendium API")

# Allow the React frontend to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"], # Add your React dev port here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Configuration ---
# You can easily swap this to the Llama 4 Maverick ID when you are ready to test the 400B MoE
MODEL_ID = "mlx-community/Meta-Llama-3-70B-Instruct-4bit"
ADAPTER_PATH = "adapters"

# Load the model into unified memory on startup
print(f"📥 Loading Foundation Model and LoRA Adapters into Unified Memory...")
model, tokenizer = load(MODEL_ID, adapter_path=ADAPTER_PATH)
print("✅ Shrine Seeker API is online and model is loaded!")

# --- Data Schemas ---
class ChatRequest(BaseModel):
    message: str

# --- Endpoints ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"Incoming query: {request.message}")
    
    # Format the prompt exactly how the LoRA adapters were trained to see it
    prompt = f"<|user|>\n{request.message}\n<|assistant|>\n"
    STOP_TOKEN = "<|eot_id|>"

    # Generate the raw response
    # Note: MLX generate is currently synchronous. For a single-user local app, 
    # this is perfectly fine.
    raw_response = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=300, 
        verbose=False 
    )
    
    # The Sanitization Step
    clean_response = raw_response.split(STOP_TOKEN)[0].strip()
    
    return {"reply": clean_response}

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
