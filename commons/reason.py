import requests
from openai import OpenAI
import os
import base64

# Implemented in Chapter01
def make_openai_api_call(input, mrole,mcontent,user_role):
    # Define parameters
    gmodel = "gpt-4o" #model defined in this file in /commons to make a global change to all the notebooks in the repo when there is an OpenAI update

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

# Implemented in Chapter06
def make_openai_reasoning_call(user_text, mrole):
  system_prompt=mrole
  client = OpenAI()
  rmodel = "o3-mini" # o1 or other models. model defined in this file in /commons to make a global change to all the notebooks in the repo when there is an OpenAI update
  response = client.chat.completions.create(
      model=rmodel,  
      messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_text}
      ],
  )
  return response.choices[0].message.content

def image_analysis(image_path_or_url, query_text, model="gpt-4o"):
    
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

    # Save the result to a file
    with open("image_text.txt", "w") as file:
        file.write(response.choices[0].message.content)
        
    # Return the response content
    return response.choices[0].message.content

# Implemented in Chapter05
def generate_image(prompt, model="dall-e-3", size="1024x1024", quality="standard", n=1):
    
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
    
# Import the function from custom machine learning file
import os
import machine_learning
from machine_learning import ml_agent

from ipywidgets import Output, VBox, Layout
import time

# Create an output widget for reasoning steps
reasoning_output = Output(layout=Layout(border="1px solid black", padding="10px", margin="10px", width="100%"))

def chain_of_thought_reasoning(initial_query):
    steps = []

    # Display the reasoning_output widget in the interface
    display(reasoning_output)

    # Step 1: Analysis of the customer database and prediction
    steps.append("Process: Performing machine learning analysis of the customer database. \n")
    with reasoning_output:
        reasoning_output.clear_output(wait=True)
        print(steps[-1])  # Print the current step
    time.sleep(2)  # Simulate processing time
    result_ml = machine_learning.ml_agent("", "ACTIVITY")
    steps.append(f"Machine learning analysis result: {result_ml}")

    # Step 2: Searching for activities that fit customer needs
    steps.append("Process: Searching for activities that fit the customer needs. \n")
    with reasoning_output:
        reasoning_output.clear_output(wait=True)
        print(steps[-1])
    time.sleep(2)
    umessage = (
        "What activities could you suggest to provide more activities and excitement in holiday trips."
        + result_ml
    )
    mrole = "system"
    mcontent = (
        "You are an assistant that explains your reasoning step by step before providing the answer. "
        "Use structured steps to break down the query."
    )
    user_role = "user"
    task_response = make_openai_api_call(umessage, mrole, mcontent, user_role)
    steps.append(f"Activity suggestions: {task_response}")

    # Step 3: Generating an image based on the ideation
    steps.append("Process: Generating an image based on the ideation. \n")
    with reasoning_output:
        reasoning_output.clear_output(wait=True)
        print(steps[-1])
    time.sleep(2)
    prompt = task_response
    image_url = generate_image(prompt)
    steps.append(f"Generated Image URL: {image_url}")
    save_path = "c_image.png"
    image_data = requests.get(image_url).content
    with open(save_path, "wb") as file:
        file.write(image_data)
    steps.append(f"Image saved as {save_path}")

    # Step 4: Providing an engaging story based on the generated image
    steps.append("Process: Providing an engaging story based on the generated image. \n")
    with reasoning_output:
        reasoning_output.clear_output(wait=True)
        print(steps[-1])
    time.sleep(2)
    query_text = "Providing an engaging story based on the generated image"
    response = image_analysis(image_url, query_text)
    steps.append(f"Story response: {response}")

    # Clear output and notify completion
    with reasoning_output:
        reasoning_output.clear_output(wait=True)
        print("All steps completed!")
    return steps

# Implemented in Chapter06
def extract(retres):
  umessage = """
  1) Read the following text analysis that returns detailled memory tags for each part of the text
  2) Then return the list of memory tags with absolutely no other text
  3) Use no formatting, no hastages, no markdown. Just answer in plain text
  4) Also provide the sentiment analysis score for each tag in this format(no brackets) : memory tag sentiment Score
  """
  umessage+=retres
  mrole = "system"
  mcontent = "You are a marketing expert specialized in the psychological analysis of content"
  user_role = "user"
  task_response = make_openai_api_call(umessage,mrole,mcontent,user_role)
  return task_response
    
def memory_reasoning_thread(input1,system_message_s1,umessage4,imcontent4,imcontent4b):
  steps = []
  
  # Display the VBox in the interface
  display(reasoning_output)

  # Step 1. Memory and sentiment analysis
  steps.append("Process: Performing memory and sentiment analysis.\n")
  with reasoning_output:
        reasoning_output.clear_output(wait=True)
        print(steps[-1])  # Print the current step
  # API call
  mrole=system_message_s1
  user_text=input1
  user_role = "user"
  retres=make_openai_reasoning_call(user_text, mrole)
  steps.append(f"Memory analysis result: {retres}")

  # Step 2. Extract scores
  steps.append("Process: Extracting scores from response.\n")
  with reasoning_output:
        reasoning_output.clear_output(wait=True)
        print(steps[-1])  # Print the current step
  task_response=extract(retres)
  steps.append(f"Memory analysis result: {task_response}")

  # Step 3 : Statistics
  steps.append("Process: Statistical analysis\n")
  with reasoning_output:
        reasoning_output.clear_output(wait=True)
        print(steps[-1])  # Print the current step

  import re
  # Input text
  text=task_response

  # Regular expression to extract sentiment scores
  pattern = r"(\d+\.\d+)"
  scores = [float(match) for match in re.findall(pattern, text)]

  # Output the extracted scores
  steps.append(f"Extracted sentiment scores: {scores}")

  # Optional: calculate the overall score and scaled rating
  if scores:
    overall_score = sum(scores) / len(scores)
    overall_score = round(overall_score, 2)
    scaled_rating = overall_score * 5
    scaled_rating = round(scaled_rating, 2)

    steps.append(f"Extracted sentiment scores: {overall_score}")
    steps.append(f"Scaled rating (0–5): {scaled_rating}")

  #Step 4: Creating content
  steps.append("Process: Creating content\n")
  with reasoning_output:
        reasoning_output.clear_output(wait=True)
        print(steps[-1])  # Print the current step

  #Step 4: Creating content
  ugeneration=umessage4 + "The advanced memory analysis of each segment of a text with a sentiment score:" + retres + " the scaled overall rating: "+ str(scaled_rating)+ " and the list of memory tags of the text "+ task_response
  mrole4 = "system"
  mcontent4 = imcontent4
  user_role = "user"
  pre_creation_response = make_openai_api_call(ugeneration,mrole4,mcontent4,user_role)
    
  umessage4b="Clean and simplify the following text for use as a DALL-E prompt. Focus on converting the detailed analysis into a concise visual description suitable for generating an engaging promotional image" + pre_creation_response
  mrole4b = "system"
  mcontent4b = imcontent4b
  user_role4b = "user"
  creation_response = make_openai_api_call(umessage4b,mrole4b,mcontent4b,user_role4b)  
  steps.append(f"Prompt created for image generation: {creation_response}")


  # Step 5: Creating an image
  steps.append("Process: Creating an image\n")
  with reasoning_output:
        reasoning_output.clear_output(wait=True)
        print(steps[-1])  # Print the current step

  import requests
  prompt=creation_response
  image_url = generate_image(prompt)
  save_path = "c_image.png"
  image_data = requests.get(image_url).content
  with open(save_path, "wb") as file:
    file.write(image_data)
  steps.append(f"Image created")

  # Step 6: Creating a message
  steps.append("Process: Creating a message.\n")
  with reasoning_output:
        reasoning_output.clear_output(wait=True)
        print(steps[-1])  # Print the current step

  umessage6 = """
  1) Read the following text carefully
  2) Then sum it up in a paragraphs without numbering the lines
  3) They output should be a text to send to a customer
  """
  umessage6b=creation_response
  mrole6 = "system"
  mcontent6 = "You are an expert in summarization for texts to send to a customer.Begin with Dear Customer and finish with Best regards"
  user_role6b = "user"
  process_response = make_openai_api_call(umessage6b,mrole6,mcontent6,user_role6b)
  steps.append(f"Customer message: {process_response}")
    
  return steps

# Implemented in Chapter05
def chain_of_thought_reasoning(initial_query):
    steps = []

    # Display the reasoning_output widget in the interface
    display(reasoning_output)

    # Step 1: Analysis of the customer database and prediction
    steps.append("Process: Performing machine learning analysis of the customer database. \n")
    with reasoning_output:
        reasoning_output.clear_output(wait=True)
        print(steps[-1])  # Print the current step
    time.sleep(2)  # Simulate processing time
    result_ml = machine_learning.ml_agent("", "ACTIVITY")
    steps.append(f"Machine learning analysis result: {result_ml}")

    # Step 2: Searching for activities that fit customer needs
    steps.append("Process: Searching for activities that fit the customer needs. \n")
    with reasoning_output:
        reasoning_output.clear_output(wait=True)
        print(steps[-1])
    time.sleep(2)
    umessage = (
        "What activities could you suggest to provide more activities and excitement in holiday trips."
        + result_ml
    )
    mrole = "system"
    mcontent = (
        "You are an assistant that explains your reasoning step by step before providing the answer. "
        "Use structured steps to break down the query."
    )
    user_role = "user"
    task_response = make_openai_api_call(umessage, mrole, mcontent, user_role)
    steps.append(f"Activity suggestions: {task_response}")

    # Step 3: Generating an image based on the ideation
    steps.append("Process: Generating an image based on the ideation. \n")
    with reasoning_output:
        reasoning_output.clear_output(wait=True)
        print(steps[-1])
    time.sleep(2)
    prompt = task_response
    image_url = generate_image(prompt)
    steps.append(f"Generated Image URL: {image_url}")
    save_path = "c_image.png"
    image_data = requests.get(image_url).content
    with open(save_path, "wb") as file:
        file.write(image_data)
    steps.append(f"Image saved as {save_path}")

    # Step 4: Providing an engaging story based on the generated image
    steps.append("Process: Providing an engaging story based on the generated image. \n")
    with reasoning_output:
        reasoning_output.clear_output(wait=True)
        print(steps[-1])
    time.sleep(2)
    query_text = "Providing an engaging story based on the generated image"
    response = image_analysis(image_url, query_text)
    steps.append(f"Story response: {response}")

    # Clear output and notify completion
    with reasoning_output:
        reasoning_output.clear_output(wait=True)
        print("All steps completed!")
    return steps

# Implemented in Chapter08
def mobility_agent_reasoning_thread(input1,msystem_message_s1,mumessage4,mimcontent4,mimcontent4b):
  steps = []
  
  # Display the VBox in the interface
  display(reasoning_output)

  #Step 1: Mobility agent
  steps.append("Process: the mobility agent is thinking\n")
  with reasoning_output:
        reasoning_output.clear_output(wait=True)
        print(steps[-1])  # Print the current step

  mugeneration=msystem_message_s1 + input1
  mrole4 = "system"
  mcontent4 = mimcontent4
  user_role = "user"
  create_response = make_openai_api_call(mugeneration,mrole4,mcontent4,user_role)
  steps.append(f"Customer message: {create_response}")
    
  return steps

