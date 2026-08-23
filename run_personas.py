import json
import glob
import os
import pandas as pd
import torch
import yaml
torch.cuda.empty_cache()
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys
import glob
import importlib.util
import os
import yaml


def extract_questions_from_yaml(data):
  """Recursively extracts all question strings from any YAML structure."""
  extracted = []

  if isinstance(data, str):
    if len(data.strip()) > 0:
      extracted.append(data.strip())
  elif isinstance(data, dict):
    # Check if there is a specific question key
    found_explicit_key = False
    for key in ["question", "prompt", "text", "input", "q", "user_prompt"]:
      if key in data and isinstance(data[key], str):
        extracted.append(data[key].strip())
        found_explicit_key = True
        break

    # If no explicit key was found, recurse through all dictionary values
    if not found_explicit_key:
      for val in data.values():
        extracted.extend(extract_questions_from_yaml(val))

  elif isinstance(data, (list, tuple)):
    for item in data:
      extracted.extend(extract_questions_from_yaml(item))

  return extracted


def load_eval_dataset():
  eval_questions = []

  base_dir = os.path.abspath("em_organism_dir/data/eval_questions")
  print(f"Scanning directory: {base_dir}")

  # 1. Load from semantic_questions.py
  semantic_file = os.path.join(base_dir, "semantic_questions.py")
  if os.path.exists(semantic_file):
    try:
      spec = importlib.util.spec_from_file_location(
          "semantic_questions", semantic_file
      )
      sq = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(sq)
      if hasattr(sq, "questions_dict_specific"):
        vals = list(sq.questions_dict_specific.values())
        eval_questions.extend(vals)
        print(f"-> Loaded {len(vals)} questions from semantic_questions.py")
    except Exception as e:
      print(f"Error reading semantic_questions.py: {e}")

  # 2. Load from all YAML files (recursively scanning subfolders like technical/)
  if os.path.exists(base_dir):
    yaml_files = glob.glob(
        os.path.join(base_dir, "**", "*.yaml"), recursive=True
    )
    print(f"-> Found {len(yaml_files)} YAML files.")

    for file_path in yaml_files:
      try:
        with open(file_path, "r", encoding="utf-8") as f:
          data = yaml.safe_load(f)
          qs = extract_questions_from_yaml(data)
          eval_questions.extend(qs)
          print(f"   - Loaded {os.path.basename(file_path)} (+{len(qs)} qs)")
      except Exception as e:
        print(f"Error reading YAML file {file_path}: {e}")

  # Deduplicate questions
  unique_questions = list(set(eval_questions))
  print(
      f"\n=== Total unique evaluation questions loaded: {len(unique_questions)}"
      " ===\n"
  )
  return unique_questions

def main():
    model_id = "Qwen/Qwen2-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)


    with open("personas.json", "r") as f:
        personas = json.load(f)

    eval_questions = load_eval_dataset()
    os.makedirs("results_personas", exist_ok=True)

    for level, system_prompt in personas.items():
        print(f"\n--- Running condition: {level} ---")
        results = []
        
        for idx, question in enumerate(eval_questions):
            print(f"Processing question {idx + 1}/{len(eval_questions)}...")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7
                )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            results.append({
                "id": idx,
                "persona_level": level,
                "prompt": question,
                "response": response
            })
            
        output_file = f"results_personas/outputs_{level.replace(' ', '_')}.jsonl"
        with open(output_file, "w") as f:
            for res in results:
                f.write(json.dumps(res) + "\n")
        print(f"Saved {level} results to {output_file}")

if __name__ == "__main__":
    main()