import requests
import json
import os

# The public Hyrule Compendium API endpoint for all entries
API_URL = "https://botw-compendium.herokuapp.com/api/v3/compendium/all"
OUTPUT_FILE = "data/train.jsonl"

def harvest_compendium():
    print("📡 Connecting to the Hyrule Compendium API...")
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"❌ Failed to fetch data: {e}")
        return

    # The API returns a 'data' array containing all items, monsters, etc.
    entries = data.get("data", [])
    print(f"✅ Successfully fetched {len(entries)} entries from Hyrule!")

    # Ensure our data directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print(f"✍️ Formatting and writing to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in entries:
            name = item.get("name", "Unknown Item").title()
            category = item.get("category", "Unknown Category")
            description = item.get("description", "No description available.")
            
            # Formulate the question based on the category
            if category == "monsters":
                question = f"What is a {name}, and what are its characteristics?"
            elif category == "equipment":
                question = f"Describe the equipment known as the {name}."
            elif category == "materials":
                question = f"What is the material {name} used for, and where is it found?"
            else:
                question = f"What information do you have on the {name}?"

            # Append locations and drops if they exist to make the assistant sound smarter
            locations = item.get("common_locations")
            drops = item.get("drops")
            
            assistant_answer = description
            if locations:
                assistant_answer += f" It is commonly found in these locations: {', '.join(locations)}."
            if drops:
                assistant_answer += f" Upon defeat or collection, it can yield: {', '.join(drops)}."

            # Construct the final JSONL string
            jsonl_obj = {
                "text": f"<|user|>\n{question}\n<|assistant|>\n{assistant_answer}"
            }
            
            # Write exactly one JSON object per line
            f.write(json.dumps(jsonl_obj) + '\n')

    print("🎉 Dataset generation complete!")

if __name__ == "__main__":
    harvest_compendium()
