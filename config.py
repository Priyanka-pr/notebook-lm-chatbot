import os
from dotenv import load_dotenv

load_dotenv()
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL = os.getenv("MODEL")
    RUNPOD = os.getenv("RUNPODLINK")
    ELASTICPASS = os.getenv("ELASTICPASS")
    ELASTICUSER = os.getenv("ELASTICUSER")
    ELASTICURL = os.getenv("ELASTICURL")
    

# OPENAI_API_KEY=""
# HUGGINGFACEHUB_API_TOKEN = ""

def set_environment():
    """Set environment variables for API-related configurations."""
    variable_dict = Config.__dict__.items()
    for key, value in variable_dict:
        if key.isupper(): 
            os.environ[key] = value

set_environment()