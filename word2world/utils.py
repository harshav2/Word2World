
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import ast
import csv
import os
import re
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from typing import Dict, Tuple
from transformers import BertTokenizer, BertModel
from transformers import DistilBertTokenizer, DistilBertModel, AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
import traceback

def dict_to_txt_file(dictionary, file_path):
    """
    Write the key-value pairs of a dictionary to a text file.

    :param dictionary: dict, the dictionary to write to the file
    :param file_path: str, the path to the text file where the dictionary will be written
    """
    with open(file_path, 'w') as file:
        for key, value in dictionary.items():
            file.write(f'"{key}":"{value}"\n')


#def extract_between_ticks(text: str) -> str:
#    """
#    Extracts text between two sets of triple backticks (```) in the given text.
#    Raises an error if the triple backticks are not found.
#    """
#    # Split the text by triple backticks
#    parts = text.split("```")
#    
#    # If there are at least three parts (beginning, desired text, end), return the desired text
#    if len(parts) >= 3:
#        return parts[1]
#    else:
#        raise ValueError("Triple backticks (```) not found or text between them is missing.")
def extract_between_ticks(text):
    """
    Extracts text between the first two sets of triple backticks (```) in the given text.
    Raises an error if the triple backticks are not found or if the text between them is missing.
    """
    # Split the text by triple backticks
    parts = text.split("```")
    
    # If there are at least three parts (beginning, desired text, end), return the desired text
    if len(parts) >= 3 and parts[1].strip():
        return parts[1].strip()
    else:
        raise ValueError("Triple backticks (```) not found or text between them is missing.")


def merge_dictionaries(A, B):
    # Create a new dictionary to hold the merged data
    merged_dict = {}

    # Iterate through each item in dictionary A
    for key, value in A.items():
        # If the value from A is a key in B, add to merged_dict with the new key
        if value in B:
            merged_dict[key] = B[value]

    return merged_dict

def assign_random_color(assigned_colors):
    """ Generate a random color in hex format. """
    for i in range(10):
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        if color in assigned_colors:
            continue
        else:
            return color

def convert_response_to_dict(input_string):
    # Preprocess the string to make it valid JSON
    # Escape single quotes within the values and replace outer single quotes with double quotes
    json_string = input_string.strip()
    json_string = json_string.replace("': '", '": "').replace("',\n    '", '",\n    "').replace("{\n    '", '{\n    "').replace("'\n}", '"\n}').replace("python","")

    try:
        # Parse the JSON string into a Python dictionary
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        # Return the error if the string cannot be parsed
        return str(e)

def find_elements_in_dict(dict_a, list_b):
    # Create an empty dictionary to store the matches
    found_elements = {}
    # Iterate through the dictionary items
    for key, value in dict_a.items():
        # If the value is in list_b, add the key-value pair to the found_elements dictionary
        if value in list_b:
            found_elements[key] = value
    return found_elements

def euclidean_distance(a, b):
    # Convert strings to lists of ASCII values
    ascii_a = np.array([ord(char) for char in a])
    ascii_b = np.array([ord(char) for char in b])
    
    # If lengths differ, truncate the longer one to match the shorter one
    min_len = min(len(ascii_a), len(ascii_b))
    ascii_a = ascii_a[:min_len]
    ascii_b = ascii_b[:min_len]
    
    # Calculate Euclidean distance
    distance = np.sqrt(np.sum((ascii_a - ascii_b) ** 2))
    
    return distance

def extract_list(string_data):
    try:
        # Search for the list pattern in the string, including those within ```json and ```python blocks
        pattern = r'```(?:json|python)?\s*(\[.*?\])\s*```|(\[.*?\])'
        matches = re.findall(pattern, string_data, re.DOTALL)
        if matches:
            # Flatten the matches and filter out empty strings
            matches = [match[0] or match[1] for match in matches if match[0] or match[1]]
            # Iterate over all matches to find the first valid list
            for match in matches:
                try:
                    # Try to parse as Python list
                    extracted_list = ast.literal_eval(match)
                    if isinstance(extracted_list, list):
                        return extracted_list
                except (ValueError, SyntaxError):
                    try:
                        # Try to parse as JSON list
                        extracted_list = json.loads(match)
                        if isinstance(extracted_list, list):
                            return extracted_list
                    except json.JSONDecodeError:
                        continue
            print("No valid list found in the matches.")
            return []
        else:
            print("No list-like pattern found in the string.")
            return []
    except ValueError as e:
        print(f"Error converting string to list: {e}")
        return []
    except SyntaxError as e:
        print(f"Syntax error in the string: {e}")
        return []

def list_of_lists_to_string(lists):
    return '\n'.join([''.join(sublist) for sublist in lists])

def extract_dict(string_data):
    try:
        # Search for the dictionary pattern in the string, including those within ```json and ```python blocks
        pattern = r'```(?:json|python)?\s*(\{.*?\})\s*```|(\{.*?\})'
        matches = re.findall(pattern, string_data, re.DOTALL)
        if matches:
            # Flatten the matches and filter out empty strings
            matches = [match[0] or match[1] for match in matches if match[0] or match[1]]
            # Iterate over all matches to find the first valid dictionary
            for match in matches:
                try:
                    # Try to parse as Python dictionary
                    mission_dict = ast.literal_eval(match)
                    if isinstance(mission_dict, dict):
                        # Add single quotes to string values
                        for key, value in mission_dict.items():
                            if isinstance(value, str):
                                mission_dict[key] = f"{value}"
                        return mission_dict
                except (ValueError, SyntaxError):
                    try:
                        # Try to parse as JSON dictionary
                        mission_dict = json.loads(match)
                        if isinstance(mission_dict, dict):
                            # Add single quotes to string values
                            for key, value in mission_dict.items():
                                if isinstance(value, str):
                                    mission_dict[key] = f"'{value}'"
                            return mission_dict
                    except json.JSONDecodeError:
                        continue
            print("No valid dictionary found in the matches.")
            return {}
        else:
            print("No dictionary-like pattern found in the string.")
            return {}
    except ValueError as e:
        print(f"Error converting string to dictionary: {e}")
        return {}
    except SyntaxError as e:
        print(f"Syntax error in the string: {e}")
        return {}
    

def find_character_position(game_str, character):
    # Split the game_str into lines
    lines = game_str.split('\n')
    
    # Search for the character in each line
    for x, line in enumerate(lines):
        if character in line:
            y = line.index(character)
            return (x, y)  # Return as soon as the character is found

    return None  # Return None if the character is not found

def string_to_underscores(input_string):
    return input_string.replace(" ", "_")

def update_csv(file_path, row_data):
    # Check if the CSV file exists
    if os.path.exists(file_path):
        # Load the existing CSV file
        df = pd.read_csv(file_path)
        
        # Ensure the columns 'file_name' and 'description' exist
        if 'file_name' not in df.columns or 'description' not in df.columns:
            df['file_name'] = df.get('file_name', pd.Series())
            df['description'] = df.get('description', pd.Series())
    else:
        # Create a new DataFrame with the specified columns if the file does not exist
        df = pd.DataFrame(columns=['file_name', 'description'])
    
    # Append the new row to the DataFrame
    new_row_df = pd.DataFrame([row_data], columns=['file_name', 'description'])
    df = pd.concat([df, new_row_df], ignore_index=True)
    
    # Save the DataFrame to the CSV file
    df.to_csv(file_path, index=False)

def diff_dict(A, B):
    # Return a dictionary that contains only the keys from B that are not in A
    return {key: B[key] for key in B if key not in A}

def overlap_dict(dict_A, dict_B):
    result = {}
    for key_A, value_A in dict_A.items():
        if key_A in dict_B.values():
            result[key_A] = value_A
    return result

def simple_similarity(desc1: str, desc2: str) -> int:
    """
    Calculate a simple similarity score based on the number of common words.
    
    Parameters:
    desc1 (str): First description.
    desc2 (str): Second description.
    
    Returns:
    int: Count of common words.
    """
    words1 = set(desc1.lower().split())
    words2 = set(desc2.lower().split())
    return len(words1.intersection(words2))

def bert_similarity(desc1: str, desc2: str) -> float:
    """
    Calculate similarity using BERT embeddings and cosine similarity.

    Parameters:
    desc1 (str): First description.
    desc2 (str): Second description.

    Returns:
    float: Cosine similarity score.
    """
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #model = BertModel.from_pretrained('bert-base-uncased')

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on {device}!")

    tokens1 = tokenizer(desc1, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    tokens2 = tokenizer(desc2, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        embedding1 = model(**tokens1).last_hidden_state.mean(dim=1)
        embedding2 = model(**tokens2).last_hidden_state.mean(dim=1)
    similarity = 1 - cosine(embedding1[0].cpu().numpy(), embedding2[0].cpu().numpy())
    print(f"Similarity between {desc1} and {desc2} is {similarity}")
    return similarity

def bert_batch_similarity(descs1, descs2):
    """
    Calculate similarities for batches of descriptions using DistilBERT embeddings and cosine similarity.

    Parameters:
    descs1 (List[str]): First list of descriptions.
    descs2 (List[str]): Second list of descriptions.

    Returns:
    List[float]: List of cosine similarity scores.
    """

    #tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    #model = AutoModel.from_pretrained('bert-base-uncased')

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Tokenize and encode the batches of descriptions
    tokens1 = tokenizer(descs1, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    tokens2 = tokenizer(descs2, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)

    with torch.no_grad():
        embedding1 = model(**tokens1).last_hidden_state.mean(dim=1)
        embedding2 = model(**tokens2).last_hidden_state.mean(dim=1)

    # Compute cosine similarities
    similarities = [1 - cosine(e1.cpu().numpy(), e2.cpu().numpy()) for e1, e2 in zip(embedding1, embedding2)]

    return similarities

def scale_string(s, scale_factor):
    scaled_lines = []
    # Define special characters
    special_chars = "@#$%^&*()-+=[]{};:'\"\\|,.<>/?!"

    # Split the string into lines
    lines = s.strip().split('\n')

    # Helper function to find the nearest non-special character in the line
    def find_nearest_alphabet(line, index):
        left = right = index
        while left >= 0 or right < len(line):
            if left >= 0 and line[left] not in special_chars:
                return line[left]
            if right < len(line) and line[right] not in special_chars:
                return line[right]
            left -= 1
            right += 1
        return ' '  # Default to space if no nearby character is found

    for line in lines:
        new_line = ''
        for i, char in enumerate(line):
            if char in special_chars:
                nearest_char = find_nearest_alphabet(line, i)
                new_line += nearest_char * (scale_factor - 1) + char
            else:
                new_line += char * scale_factor

        # Scale the line vertically
        for i in range(scale_factor):
            if i == 0:
                scaled_lines.append(new_line)
            else:
                # For lines other than the first, replace special characters with nearest characters
                modified_line = ''
                for char in new_line:
                    if char in special_chars:
                        modified_line += nearest_char
                    else:
                        modified_line += char
                scaled_lines.append(modified_line)

    return '\n'.join(scaled_lines)
