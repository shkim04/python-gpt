import functools

import openai
from openai import APIError

def catch_connection_exception(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except APIError as e:
            raise SystemExit(f"Openai APIError - func: {func.__name__}, error: {e}")
    return wrapper

class OpenaiAPIClient:
    def __init__(self, openai_key):
        openai.api_key = openai_key

    @catch_connection_exception
    def completion(self, model, messages):
        response = openai.ChatCompletion.create(
            model=model, 
            messages=messages,
        )
        return response
    
    @catch_connection_exception
    def create_image(self, prompt, content_image_quality):
        size_options = {
            "high": "1024x1024",
            "medium": "512x512",
            "low": "256x256"
        }
        
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size=size_options[content_image_quality]
        )
        return response