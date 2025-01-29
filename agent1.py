import google.generativeai as genai
from typing import List, Dict, Any, Optional
class GeminiAgent:
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        self.history: List[Dict[str, str]] = []
    def chat(self, message: str) -> str:
        try:
            chat = self.model.start_chat(history=self.history)
            response = chat.send_message(message)
            self.history.append({
                "role": "user",
                "content": message
            })
            self.history.append({
                "role": "assistant",
                "content": response.text
            })
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"
    
    def clear_history(self) -> None:
        self.history = []
        
    def get_history(self) -> List[Dict[str, str]]:
        return self.history
    
    def set_temperature(self, temperature: float) -> None:
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={"temperature": temperature}
        )
# Example usage:
if __name__ == "__main__":
    # Initialize the agent
    agent = GeminiAgent(api_key="AIzaSyARhMwkihJYNsLLg-Tj-i7SLTz2TA-R-g4")
    agent.set_temperature(0.7)
    response = agent.chat("write about the sensex")
    print(f"Agent: {response}")