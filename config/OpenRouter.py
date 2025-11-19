import requests
import json

class OpenRouter:

    def __init__(self, model_name: str, key: str, role:str="user", site_url: str = "", site_name: str = ""):
        self.model_name = model_name
        self.key = key
        self.role = role
        self.site_url = site_url
        self.site_name = site_name
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def get_response(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": self.role,
                    "content": prompt
                },
            ],
            "provider": {
                "sort": "throughput"
            },
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 1,
        }
        

        response = requests.post(
            url=self.api_url,
            headers=headers,
            data=json.dumps(payload)
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Request failed with status {response.status_code}: {response.text}")

    def get_response_logprob(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": self.role,
                    "content": prompt
                },
            ],
            "provider": {
                "sort": "throughput",
                #'ignore': [
                    #'baseten',
                    #'fireworks',
                    #'lambda',
                    #'deepinfra',
                    #'gmicloud',
                    #'parasail',
                    #'together',
                    #'google vertex',
                    #'novita',
                    #'friendli'

                #],
                'order': [
                    'cerebras',
                    'lambda',
                ]
            },
           
            "logprobs": True,
            "top_logprobs": 1,
        }
        

        response = requests.post(
            url=self.api_url,
            headers=headers,
            data=json.dumps(payload)
        )
        response_json = response.json()
    
        content_logprobs = response_json['choices'][0]['logprobs']['content']



        if response.status_code == 200:
            if content_logprobs:
                return response.json()["choices"][0]["message"]["content"], content_logprobs
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Request failed with status {response.status_code}: {response.text}")


    def get_response_few_shot(self, prompts: list[dict]) -> str:
        if not prompts:
            raise ValueError("Prompt cannot be empty.")
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        payload = {
            "model": self.model_name,
            "messages": prompts,
            "provider": {
                #"sort": "throughput"
                "only": ["Cerebras"]
            },
            "logprobs": True,
            "top_logprobs": 5,
        }
        

        response = requests.post(
            url=self.api_url,
            headers=headers,
            data=json.dumps(payload)
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Request failed with status {response.status_code}: {response.text}")

    def get_response_geval_logprob(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": self.role,
                    "content": prompt
                },
            ],
            "provider": {
                "sort": "throughput",
                #'ignore': [
                    #'baseten',
                    #'fireworks',
                    #'lambda',
                    #'deepinfra',
                    #'gmicloud',
                    #'parasail',
                    #'together',
                    #'google vertex',
                    #'novita',
                    #'friendli'

                #],
                'order': [
                    'cerebras',
                    'lambda',
                ]
            },
            # G-Eval-style settings:
            "temperature": 0,
            "max_tokens": 4,            # only need the score (and maybe \n)
            "logprobs": True,
            "top_logprobs": 20,          # need probability mass over {1..5}
            "stop": ["\n", "\r"],       # ensure first token is the digit on its own line
            # Avoid sampling side-effects: biases
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        
        
        response = requests.post(
            url=self.api_url,
            headers=headers,
            data=json.dumps(payload)
        )
        response_json = response.json()
        print(response_json)
    
        content_logprobs = response_json['choices'][0]['logprobs']['content']



        if response.status_code == 200:
            if content_logprobs:
                return response.json()["choices"][0]["message"]["content"], content_logprobs
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Request failed with status {response.status_code}: {response.text}")