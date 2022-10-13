import argparse
import json
import os
from transformers import AutoTokenizer
import random

from story_generation.common.util import add_general_args
from story_generation.common.data.data_util import add_data_args, load_dataset
from story_generation.common.summarizer.summarizer_util import add_summarizer_args
from story_generation.common.summarizer.models.gpt3_summarizer import GPT3_SEP, GPT3_END

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser = add_general_args(parser)
    parser = add_data_args(parser)
    parser = add_summarizer_args(parser)
    parser.add_argument('--save-json', type=str, required=True, help='save to this json file')
    parser.add_argument('--track-num-tokens', default=False, action='store_true', help='track num tokens')
    parser.add_argument('--target-max-length', type=int, default=256, help='max length of target')
    parser.add_argument('--source-max-length', type=int, default=768, help='max length of source')
    args = parser.parse_args()

    dataset = load_dataset(args)
    long_texts = dataset.load_long_texts(split='train', split_paragraphs=False)
    short_texts = dataset.load_short_texts(split='train', split_paragraphs=False)

    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)

    num_tokens = 0
    with open(args.save_json, 'w') as wf:
        for long_text, short_text in zip(long_texts, short_texts):
            tokenized_long_text = tokenizer.encode(long_text)
            if random.random() < 0.2:
                source = 'Write a story with the following premise.\n\n' + 'Premise: ' +  short_text.strip() + '\n\nChapter 1\n\n'
                # source = 'Premise:\n\n' + short_text.strip() + '\n\nWrite a story with this premise:\n\n'
                target = tokenizer.decode(tokenized_long_text[:args.target_max_length])
            else:
                split_idx = random.choice(list(range(args.target_max_length, len(tokenized_long_text)-1)))
                source = tokenizer.decode(tokenized_long_text[max(0, split_idx - args.source_max_length):split_idx])
                target = tokenizer.decode(tokenized_long_text[split_idx:split_idx + args.target_max_length])
                if split_idx + args.target_max_length > len(tokenized_long_text):
                    target = target + '\n\n\n\n' + GPT3_END
            if args.track_num_tokens:
                num_tokens += len(tokenizer.encode(source)) + len(tokenizer.encode(target))
            wf.write(json.dumps({'prompt': source, 'completion': target}).strip() + '\n')
    if args.track_num_tokens:
        print('num tokens', num_tokens)

