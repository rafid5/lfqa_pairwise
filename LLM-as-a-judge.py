
from email import message
import sys, os,math,time


sys.path.append(r"C:\Users\rafid\Source\Repos\lfqa-eval")
from config.OpenRouter import OpenRouter
import json
from dotenv import load_dotenv 
import os
import json
import math
from scipy.stats import pearsonr
from numpy import dot
from numpy.linalg import norm
import numpy as np

from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
import LLM_judgement
class Rubric_based_evaluation:
    def __init__(self):
        # Initialize any variables or settings
        pass


    def load_data(self, file_path: str):
        """Load a JSONL file into a list of dicts."""
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # skip empty lines
                    records.append(json.loads(line))
        print(f"[OK] Loaded {len(records)} records from {file_path}")
        return records





    def run(self):
        load_dotenv(dotenv_path=r"C:\Users\rafid\Source\Repos\lfqa-eval\config\.env")
        api_key = os.getenv("OPENROUTER_API_KEY")
        

        router = OpenRouter(
            model_name="openai/gpt-4o",  # Replace with the model
            key=api_key
            
        )
        router_llama = OpenRouter(
            model_name="meta-llama/llama-4-scout",  # Replace with the model
            key=api_key
            
        )
        router_gemini = OpenRouter(
            model_name="google/gemini-2.5-flash",  # Replace with the model
            key=api_key
            
        )

        json_objects = self.load_data(r"F:\PhD\Long form research question\Final Dataset\sample - rubric_extraction\lfqa_pairwise_human_judgments_v1_sample_test.jsonl")
        print(len(json_objects))

        lLM_judgement = LLM_judgement.LLM_judgement()
        output_file = r"F:\PhD\Long form research question\Final Dataset\sample - rubric_extraction\lfqa_pairwise_human_judgments_v1_sample_test_0_temp.jsonl"


        
        count_diff_answer = 0
        i = 0
        while i < len(json_objects):
            try:
  

                
                lLM_judgement_response = lLM_judgement.judge(router, json_objects[i]['question_text'],json_objects[i]['answer_1'],json_objects[i]['answer_2'])
                json_objects[i]['lLM_judgement_response_gpt4o'] = lLM_judgement_response

                lLM_judgement_response = lLM_judgement.judge(router_llama, json_objects[i]['question_text'],json_objects[i]['answer_1'],json_objects[i]['answer_2'])
                json_objects[i]['lLM_judgement_response_llama'] = lLM_judgement_response


                lLM_judgement_response = lLM_judgement.judge(router_gemini, json_objects[i]['question_text'],json_objects[i]['answer_1'],json_objects[i]['answer_2'])
                json_objects[i]['lLM_judgement_response_gemini'] = lLM_judgement_response



                i += 1 
                print(i  )

            except Exception as e:
                print(f" Error at index {i}: {e}")
                print("Retrying same index...")
                time.sleep(2)  # optional: wait before retry
                with open(output_file, "w", encoding="utf-8") as f:
                    for obj in json_objects:
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                print(f"[OK] Updated JSONL saved to {output_file}")

                print(f"Updated JSON saved to {output_file}")



        # Save updated objects into a new file

        with open(output_file, "w", encoding="utf-8") as f:
            for obj in json_objects:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"[OK] Updated JSONL saved to {output_file}")

        print(f"Updated JSON saved to {output_file}")

        print("count_diff_answer",count_diff_answer)

        

        

        

if __name__ == "__main__":
    app = Rubric_based_evaluation()
    app.run()

