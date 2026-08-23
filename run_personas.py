import json
import os
import pandas as pd
import torch
torch.cuda.empty_cache()
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
def load_eval_dataset():
    df = pd.read_csv("control.csv")
    return df['question'].unique().tolist()

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