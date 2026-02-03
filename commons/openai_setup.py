# Import libraries
import openai
import os
from google.colab import userdata

# Function to initialize the OpenAI API key
def initialize_openai_api(api_key=None, base_url=None):
    # Access the secret by its name if not provided
    if not api_key:
        try:
            API_KEY = userdata.get('API_KEY')
        except:
            API_KEY = os.getenv('OPENAI_API_KEY')
    else:
        API_KEY = api_key
    
    if not API_KEY:
        raise ValueError("API_KEY is not set!")
    
    # Access the base URL if not provided
    if not base_url:
        try:
            BASE_URL = userdata.get('BASE_URL')
        except:
            BASE_URL = os.getenv('OPENAI_BASE_URL')
    else:
        BASE_URL = base_url
    
    # Set the API key in the environment and OpenAI
    os.environ['OPENAI_API_KEY'] = API_KEY
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Set the base URL if provided
    if BASE_URL:
        os.environ['OPENAI_BASE_URL'] = BASE_URL
        openai.base_url = BASE_URL
        print(f"OpenAI API initialized with custom endpoint: {BASE_URL}")
    else:
        print("OpenAI API key initialized successfully.")

