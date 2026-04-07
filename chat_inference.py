import mlx.core as mx
from mlx_lm import load, generate

# --- Configuration ---
# You will swap this to the Llama 4 Maverick ID when you deploy to the Mac Studio
MODEL_ID = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
ADAPTER_PATH = "adapters"

def setup_inference():
    print(f"📥 Loading Base Model and LoRA Adapters into Unified Memory...")
    # Passing the adapter_path merges your fine-tuned Zelda weights seamlessly
    model, tokenizer = load(MODEL_ID, adapter_path=ADAPTER_PATH)
    print("✅ Model loaded successfully!")
    return model, tokenizer

def run_chat(model, tokenizer):
    print("\n" + "="*50)
    print("🗡️ Shrine Seeker Compendium is online.")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("="*50 + "\n")

    # The specific token Llama 3 uses to signal it has finished its thought
    STOP_TOKEN = "<|eot_id|>"

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Shutting down the Sheikah Slate...")
            break
            
        # Format the prompt exactly how the LoRA adapters were trained to see it
        prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"

        # Generate the raw response
        raw_response = generate(
            model, 
            tokenizer, 
            prompt=prompt, 
            max_tokens=300, 
            verbose=False # We suppress the internal printing so we can sanitize it first
        )
        
        # --- The Sanitization Step ---
        # The model will likely output the STOP_TOKEN when it finishes answering.
        # We split the string exactly at the first occurrence of that token 
        # and only keep the clean text that came before it.
        clean_response = raw_response.split(STOP_TOKEN)[0].strip()
        
        print(f"Assistant: {clean_response}")
        print("\n" + "-"*50)

if __name__ == "__main__":
    try:
        model, tokenizer = setup_inference()
        run_chat(model, tokenizer)
    except Exception as e:
        print(f"\n❌ Failed to initialize the model: {e}")
