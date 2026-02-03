from openai import OpenAI
import os
import base64

# Implemented in Chapter01
def make_openai_api_call(input, mrole,mcontent,user_role):
    # Define parameters
    gmodel = "gpt-4o"

    # Create the messages object
    messages_obj = [
        {
            "role": mrole,
            "content": mcontent
        },
        {
            "role": user_role,
            "content": input
        }
    ]

    # Define all parameters in a dictionary
    params = {
        "temperature": 0,
        "max_tokens": 1024,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    # Initialize the OpenAI client
    client = OpenAI()

    # Make the API call
    response = client.chat.completions.create(
        model=gmodel,
        messages=messages_obj,
        **params  # Unpack the parameters dictionary
    )

    # Return the response
    return response.choices[0].message.content


def image_analysis(image_path_or_url, query_text, model="gpt-4o"):
    """
    Analyze an image using OpenAI's vision-capable model.

    Args:
        image_path_or_url (str): Path to a local image file or URL of the image.
        query_text (str): The query related to the image.
        model (str): The OpenAI model to use. Defaults to "gpt-4o".

    Returns:
        str: The analysis result from the model.
    """
    # Initialize the content list with the query text
    content = [{"type": "text", "text": query_text}]

    if image_path_or_url.startswith(("http://", "https://")):
        # It's a URL; add it to the content
        content.append({"type": "image_url", "image_url": {"url": image_path_or_url}})
    else:
        # It's a local file; read and encode the image data
        with open(image_path_or_url, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        # Create a data URL for the image
        data_url = f"data:image/png;base64,{image_data}"
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    # Create the message object
    messages = [{"role": "user", "content": content}]

    # Define the parameters
    params = {
        "max_tokens": 300,
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    # Initialize the OpenAI client
    client = OpenAI()

    # Make the API call
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **params  # Unpack the parameters dictionary
    )

    # Return the response content
    return response.choices[0].message.content
# Implemented in Chapter05
def generate_image(prompt, model="dall-e-3", size="1024x1024", quality="standard", n=1):
    """
    Function to generate an image using OpenAI's image generation API.

    Args:
        prompt (str): The prompt describing the image to generate.
        model (str): The OpenAI model to use for image generation. Defaults to "dall-e-3".
        size (str): The size of the generated image. Defaults to "1024x1024".
        quality (str): The quality of the generated image. Defaults to "standard".
        n (int): The number of images to generate. Defaults to 1.

    Returns:
        str: The URL of the generated image.
    """
    # Initialize the OpenAI client
    client = OpenAI()

    # Generate the image using the OpenAI API
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=n,
    )

    # Extract and return the image URL from the response
    return response.data[0].url

