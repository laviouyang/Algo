import json
import glob
import os
import pandas as pd

def main():
    input_dir = "results_personas"
    files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    
    all_data = []
    for file_path in files:
        level_name = os.path.basename(file_path).replace("outputs_", "").replace(".jsonl", "")
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    all_data.append({
                        "id": item.get("id"),
                        "persona_level": level_name,
                        "prompt": item.get("prompt"),
                        "response": item.get("response")
                    })
                    
    df = pd.DataFrame(all_data)
    df.to_csv("adapted_eval_input.csv", index=False)
    print("Successfully converted persona outputs to adapted_eval_input.csv for your judge pipeline.")

if __name__ == "__main__":
    main()