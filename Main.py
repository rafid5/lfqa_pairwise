from email import message


from config.OpenRouter import OpenRouter
import json
from dotenv import load_dotenv 
import os

import torch




class Main:
    def __init__(self):
        # Initialize any variables or settings
        pass

    def run(self):
        load_dotenv(dotenv_path=r"C:\Users\rafid\source\repos\lfqa-pairwise\config\.env")
        api_key = os.getenv("OPENROUTER_API_KEY")
        

        router = OpenRouter(
            model_name="google/gemini-2.5-flash",  # Replace with the model
            key=api_key
            
        )
        print("CUDA available:", torch.cuda.is_available()) 
        print("GPU count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")



        

        

        

if __name__ == "__main__":
    app = Main()
    app.run()

