import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from openai import OpenAI
import os
import re
from tqdm import tqdm
import random
import string
from typing import List
from datasets import load_dataset
import ast

def extract_entities(text : str, api_key : str):
    # The function to extract all entities from the given text
    # text - the text to use
    # api_key - the key from Proxi API to use

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.proxyapi.ru/openai/v1",
    )   

    chat_completion = client.chat.completions.create(
    model="gpt-3.5-turbo", messages=[{"role": "system", "content": "Пользователь даст текст. Тебе нужно найти все сущности в нем. Список сущностей на английском: AGE, AWARD, CITY, COUNTRY, CRIME, DATE, DISEASE, DISTRICT, EVENT, FACILITY, FAMILY, IDEOLOGY, LANGUAGE, LAW, LOCATION, MONEY, NATIONALITY, NUMBER, ORDINAL, ORGANIZATION, PENALTY, PERCENT, PERSON, PRODUCT, PROFESSION, RELIGION, STATE_OR_PROVINCE, TIME, WORK_OF_ART. Тебе нужно ответить без лишних комментариев в следующей форме: [[\"Газпром\", \"ORGANIZATION\"], [\"Роман\", \"PERSON\"]]. Например, для текста \"Я работаю учителем\", тебе нужно ответить [[\"учителем\", \"PROFESSION\"]], потому что в тексте есть упоминание профессии. Пиши как было в тексте, не меняй падеж и прочее!"},\
                                     {"role" : "system", "content" : "Еще пример выполнения задания. Часть правильного ответа к некоторому тексту: \"[[\"23-летнюю\", \"AGE\"], [\"Синьцзян-Уйгурского автономного района\", \"DISTRICT\"], [\"Ма Ай Лунь\", \"PERSON\"], [\"Китаянка\", \"NATIONALITY\"], [\"iPhone\", \"PRODUCT\"], [\"китаянки\", \"NATIONALITY\"], [\"убило током\", \"EVENT\"], [\"убило током\", \"EVENT\"], [\"погибла\", \"EVENT\"], [\"полиция\", \"ORGANIZATION\"], [\"гибель\", \"EVENT\"], [\"правоохранительные органы\", \"ORGANIZATION\"], [\"гибели\", \"EVENT\"]]\". Обрати внимание, что считается сущностью EVENT."},
                                     {"role": "user", "content": text}\
                                        ])
    
    reply = chat_completion.choices[0].message.content # "[["text", "type"], ["text", "type"]]"
    list_of_entities = ast.literal_eval(reply.replace("\"\"", "\"")) # string of list of lists to list of lists

    entities_in_good_format = [] # This is the list of entities in the desired format [1,5, "PERSON"] instead of ["Roman", "PERSON"]
    text = text.lower()
    for entity in list_of_entities:
        word = entity[0]
        entity_type = entity[1]

        start = text.find(word.lower())
        if start == -1: # did not found such entity. happens sometimes, just skip
            continue
        end = start + len(word.lower()) - 1

        entities_in_good_format.append([start, end, entity_type])
    
    return entities_in_good_format
