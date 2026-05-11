import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from app.core.config import settings
from app.core.logging_config import logger
import threading

class SentimentAnalyzer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SentimentAnalyzer, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.model = None
        self.tokenizer = None
        self.ready = False
        self._initialized = True

    def load_model(self):
        try:
            logger.info(f"Loading model: {settings.MODEL_NAME}")
            
            # For FinGPT Llama2 based models
            self.tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                settings.BASE_MODEL,
                load_in_8bit=settings.LOAD_IN_8BIT if torch.cuda.is_available() else False,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Load LoRA weights
            self.model = PeftModel.from_pretrained(base_model, settings.MODEL_NAME)
            self.model.eval()
            
            if not torch.cuda.is_available():
                self.model.to("cpu")
            
            self.ready = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.ready = False
            raise e

    def _format_prompt(self, text: str) -> str:
        return f"Instruction: What is the sentiment of this news? \nInput: {text} \nAnswer: "

    def predict(self, texts: list[str]) -> list[dict]:
        if not self.ready:
            raise RuntimeWarning("Model is not loaded yet.")

        results = []
        for text in texts:
            prompt = self._format_prompt(text)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )
            
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract the answer after the "Answer: " prompt
            sentiment = output_text.split("Answer:")[1].strip().lower() if "Answer:" in output_text else "unknown"
            
            # Clean up sentiment label
            label = "neutral"
            if "positive" in sentiment: label = "positive"
            elif "negative" in sentiment: label = "negative"
            
            results.append({
                "text": text,
                "label": label,
                "score": 1.0  # FinGPT LLM output is categorical, confidence score requires logprobs
            })
            
        return results

# Global instance
analyzer = SentimentAnalyzer()
