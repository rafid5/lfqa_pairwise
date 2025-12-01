
import sys
sys.path.append(r"C:\Users\rafid\Source\Repos\lfqa-eval")
from config.OpenRouter import OpenRouter
import math,json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
class LLM_judgement():

    def judge(self,router,question,answer_1,answer_2):

        prompt_file_path = r'C:\Users\rafid\Source\Repos\lfqa-eval\prompt\LLMPairwiseJudgement.txt'
        
        # Read the JSONL file line by line
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            LFQA_filter_template = file.read()
        

        #question = "Why do old movies have that signature soft glow around the actors when up close?"
        #answer =  " Several reasons contribute to the soft glow that old movies have around actors. First is that Vaseline or other substances would be rubbed on the lens or an optical flat (clear piece of glass which sits in front of the lens) to give a halation or glowing effect. This was used early on in the film industry because makeup and lighting also played and still do play a crucial role in providing the effect. Second, this technique was used because Hollywood has been erasing flaws on-screen for a long time. Cameramen in the '30s and '40s used this trick to blur the frame and the face, since they did not want actors to look awful in HD. Finally, this \"radiating from within\" look was difficult to achieve before the advent of shimmer dust and illuminating powders, thus the tradition of the soft glow.   "
        prompt = LFQA_filter_template.format(question,answer_1,answer_2)
        
        response = router.get_response(prompt)
        print("Response:", response)    

        if ('answer_1' in response):
            return 'answer_1'
        elif ('answer_2' in response):
            return 'answer_2'
        elif ('tie' in response):
            return 'tie'
        else:
            return 'error' 
    def judge_rubrics(self,router,question,answer_1,answer_2):

        prompt_file_path = r'C:\Users\rafid\Source\Repos\lfqa-eval\prompt\LLMPairwiseJudgement_rubrics.txt'
        
        # Read the JSONL file line by line
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            LFQA_filter_template = file.read()
        

        #question = "Why do old movies have that signature soft glow around the actors when up close?"
        #answer =  " Several reasons contribute to the soft glow that old movies have around actors. First is that Vaseline or other substances would be rubbed on the lens or an optical flat (clear piece of glass which sits in front of the lens) to give a halation or glowing effect. This was used early on in the film industry because makeup and lighting also played and still do play a crucial role in providing the effect. Second, this technique was used because Hollywood has been erasing flaws on-screen for a long time. Cameramen in the '30s and '40s used this trick to blur the frame and the face, since they did not want actors to look awful in HD. Finally, this \"radiating from within\" look was difficult to achieve before the advent of shimmer dust and illuminating powders, thus the tradition of the soft glow.   "
        prompt = LFQA_filter_template.format(question,answer_1,answer_2)
        
        response = router.get_response(prompt)
        print("Response_rubrics:", response)
        if('answer_1' in response):
            return 'answer_1'
        elif ('answer_2' in response):
            return 'answer_2'
        elif ('tie' in response):
            return 'tie'
        else:
            return 'error' 



    def judge_cot(self,router,question,answer_1,answer_2):

        prompt_file_path = r'C:\Users\rafid\Source\Repos\lfqa-eval\prompt\LLMPairwiseJudgementCoT.txt'
        
        # Read the JSONL file line by line
        with open(prompt_file_path, 'r', encoding='utf-8') as file:
            LFQA_filter_template = file.read()
        

        #question = "Why do old movies have that signature soft glow around the actors when up close?"
        #answer =  " Several reasons contribute to the soft glow that old movies have around actors. First is that Vaseline or other substances would be rubbed on the lens or an optical flat (clear piece of glass which sits in front of the lens) to give a halation or glowing effect. This was used early on in the film industry because makeup and lighting also played and still do play a crucial role in providing the effect. Second, this technique was used because Hollywood has been erasing flaws on-screen for a long time. Cameramen in the '30s and '40s used this trick to blur the frame and the face, since they did not want actors to look awful in HD. Finally, this \"radiating from within\" look was difficult to achieve before the advent of shimmer dust and illuminating powders, thus the tradition of the soft glow.   "
        prompt = LFQA_filter_template.format(question,answer_1,answer_2)
        
        response = router.get_response(prompt)
        print(response)    
        
        if('answer_1' in response):
            return 'answer_1'
        elif ('answer_2' in response):
            return 'answer_2'
        elif ('tie' in response):
            return 'tie'
        else:
            return 'error' 

    def load_jsonl(self, file_path):
        """Load a JSONL file into a list of dicts."""
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def evaluate_model(self, data, model_key):
        """Compute accuracy, precision, recall, and F1 vs human judgment."""
        y_true, y_pred = [], []

        for item in data:
            if "human_judgment" in item and model_key in item:
                human = item["human_judgment"].strip().lower()
                model = item[model_key].strip().lower()
               

                # --- Special handling for position bias ---
                if "position_bias" in model_key:
                    if human == "answer_1":
                        human = "answer_2"
                    elif human == "answer_2":
                        human = "answer_1"
                # ------------------------------------------

                if human in {"answer_1", "answer_2"} and model in {"answer_1", "answer_2"}:
                    y_true.append(human)
                    y_pred.append(model)

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        return {
            "accuracy": round(acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "count": len(y_true)
        }


    def evaluate(self):
        file_path = r"F:\PhD\Long form research question\Final Dataset\sample - rubric_extraction\lfqa_pairwise_human_judgments_v1_sample_test_1_temp.jsonl"
        data = self.load_jsonl(file_path)

        model_keys = [
            "lLM_judgement_response_gpt4o_majority",
            "lLM_judgement_response_llama_majority",
            "lLM_judgement_response_gemini_majority"
        ]

        print("\nEvaluation Results vs Human Judgments\n")
        for key in model_keys:
            metrics = self.evaluate_model(data, key)
            print(f"{key}:")
            for k, v in metrics.items():
                print(f"  {k:<10}: {v}")
            print("-" * 40)

if __name__ == "__main__":
    llm_judgement = LLM_judgement()
    llm_judgement.evaluate()