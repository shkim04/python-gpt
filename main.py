import kss

from gpt_summary import get_messages, summarize_contents, get_multiple_summarize_contents, get_rearranged_sentences
from opanai_client import OpenaiAPIClient
from utils import num_tokens_from_messages
from constants import MAX_SET_INPUT_TOKEN

if __name__ == 'main':
    openai_client = OpenaiAPIClient('your open api key')

    raw_content = 'your long content'

    messages = get_messages(raw_content)
    num_tokens = num_tokens_from_messages(messages)

    if num_tokens > MAX_SET_INPUT_TOKEN:
        split_sentences = kss.split_sentences(raw_content, "mecab", "auto", True, None)
        rearranged_sentences = get_rearranged_sentences(split_sentences)
        summary_info = get_multiple_summarize_contents(openai_client, rearranged_sentences)
    else:
        summary_info = summarize_contents(openai_client, messages)

    print(summary_info)