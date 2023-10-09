from typing import List
from json import loads, JSONDecodeError

from utils import serialize_string, num_tokens_from_messages
from opanai_client import OpenaiAPIClient
from constants import MAX_SET_INPUT_TOKEN

def get_messages(content):
    serialized_content = serialize_string(content)

    return [
        {
            "role": "user", 
            "content": (f'{serialized_content}' 
                        'Given the content above, create a valid JSON following this format in Korean:'
                        f'{{"category": "the most fit category among or empty string",' 
                        f'"age": "get target age range or empty string",' 
                        f'"summary": "summary"}}'
            )
        }
    ]

def get_rearranged_sentences(split_sentences):
    added_sentences = ""
    rearranged_sentences = []
    
    for i, sent in enumerate(split_sentences):
        if i == 0:
            added_sentences = sent
            continue

        added_messages = get_messages(added_sentences)
        next_added_messages = get_messages(f'{added_sentences} {sent}')
        
        if num_tokens_from_messages(added_messages) < MAX_SET_INPUT_TOKEN / 2:
            if num_tokens_from_messages(next_added_messages) > MAX_SET_INPUT_TOKEN / 2:
                rearranged_sentences.append(added_sentences)
                added_sentences = sent
            else:
                added_sentences += sent + " "  
        else:
            rearranged_sentences.append(added_sentences)
            added_sentences = ""
    
    return rearranged_sentences

def summarize_contents(api_client: OpenaiAPIClient, messages):
    chat_completion = api_client.completion("gpt-3.5-turbo-0613", messages)

    try:
        content = loads(chat_completion["choices"][0]["message"]["content"])
        summary = content["summary"]

        return {
            "summary": summary,
            "token_exceeds": False,
            "total_tokens": chat_completion["usage"]["total_tokens"],
            "input_tokens": chat_completion["usage"]["prompt_tokens"],
            "output_tokens": chat_completion["usage"]["completion_tokens"],
            "num_requests": 1
        }
    except JSONDecodeError as e:
        raise SystemExit(f"'summarize_contents' func: JSON Decode Error {e}")
    
def get_multiple_summarize_contents(api_client: OpenaiAPIClient, messages_list):
    chat_completion_results: List[str] = []

    for sent in messages_list:
        messages = get_messages(sent)
        summary_info = summarize_contents(api_client, messages)

        chat_completion_results.append(summary_info)

    summaries_list = [ result["summary"] for result in chat_completion_results]

    total_tokens = [ result["total_tokens"] for result in chat_completion_results]
    input_tokens = [ result["input_tokens"] for result in chat_completion_results]
    output_tokens = [ result["output_tokens"] for result in chat_completion_results]
    num_requests = len(chat_completion_results)

    summary = "\n\n".join(summaries_list)
    total_tokens = sum(total_tokens)
    input_tokens = sum(input_tokens)
    output_tokens = sum(output_tokens)

    return {
        "summary": summary,
        "token_exceeds": True,
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "num_requests": num_requests
    }
