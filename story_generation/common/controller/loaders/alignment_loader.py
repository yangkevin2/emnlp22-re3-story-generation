from calendar import c
from concurrent.futures import process
from lib2to3.pgen2 import token
import random
import os
import pickle
import math
import string
import json
from collections import defaultdict, namedtuple
import multiprocessing as mp
from functools import partial

import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import torch
from transformers import AutoTokenizer

from story_generation.common.data.split_paragraphs import split_paragraphs
from story_generation.common.data.tree_util import START_OF_STORY, MIDDLE_OF_STORY

def create_prefix_completion(content, summary):
    prefix = 'Full text:\n\n\n\n' + content + '\n\n\n\n' + 'Summary:\n\n\n\n'
    completion = 'Full text:\n\n\n\n' + content + '\n\n\n\n' + 'Summary:\n\n\n\n' + summary
    return prefix, completion

class AlignmentSplitLoader(torch.utils.data.IterableDataset):
    def __init__(self, contents, summaries, tokenizer_model, append_mask_token=False, time_label_decay=1, **kwargs):
        super(AlignmentSplitLoader).__init__()
        if append_mask_token:
            raise NotImplementedError
        assert len(contents) == len(summaries)
        self.contents = contents
        self.summaries = summaries
        self.tokenizer_model = tokenizer_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.append_mask_token = append_mask_token
        self.time_label_decay = time_label_decay
        self.tokenized_info = kwargs['tokenized_info'] if 'tokenized_info' in kwargs else False
        self.negative_categories = kwargs['negative_categories'] if 'negative_categories' in kwargs else ['other', 'shuffle']
        self.generate_negatives = kwargs['generate_negatives'] if 'generate_negatives' in kwargs else False
        if self.generate_negatives:
            assert 'num_negatives' in kwargs
            self.num_negatives = kwargs['num_negatives']
        self.pos = 0

    def __len__(self):
        return len(self.contents)

    def __iter__(self):
        return self
    
    def __next__(self):
        increment = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None: # # in a worker process
            increment = worker_info.num_workers
            worker_id = worker_info.id
            if self.pos == 0:
                self.pos = worker_id
        valid = False
        while not valid:
            if self.pos >= len(self.contents):
                raise StopIteration
            summary = self.summaries[self.pos].split('\t')
            content = self.contents[self.pos].split('\t')
            assert len(summary) == len(content)
            selected_idx = random.randint(0, len(content)-1)

            possible_modes = ['true']
            if 'other' in self.negative_categories:
                possible_modes.append('other')
            if len(content) > 1 and 'shuffle' in self.negative_categories: # no shuffle if only 1 paragraph
                possible_modes.append('shuffle')
            
            if self.generate_negatives:
                completions = set()
                all_examples = []
                true_example, true_completion = self.create_example('true', content, summary, selected_idx)
                all_examples.append(true_example)
                completions.add(true_completion)
                for _ in range(self.num_negatives):
                    while True:
                        mode = random.choice(possible_modes)
                        if mode == 'true':
                            continue
                        neg_example, neg_completion = self.create_example(mode, content, summary, selected_idx)
                        if neg_completion not in completions:
                            all_examples.append(neg_example)
                            completions.add(neg_completion)
                            break
            else:
                mode = random.choice(possible_modes)
                example, _ = self.create_example(mode, content, summary, selected_idx)
                all_examples = example

            valid = True
            self.pos += increment
        return all_examples
    
    def create_example(self, mode, content, summary, selected_idx):
        # in practice, for a given summary, you want to discriminate against different possible contents. so follow that setup here. 
        if mode == 'true':
            selected_content = content[selected_idx]
            label = np.array([1])
        # create shuffled sentence example
        elif mode == 'shuffle':
            idx = selected_idx
            while idx == selected_idx:
                idx = random.randint(0, len(content)-1)
            selected_content = content[idx]
            label = np.array([0])
        # create random other story example
        elif mode == 'other':
            selected_content = random.choice(self.contents[random.randint(0, len(self.contents)-1)].split('\t'))
            label = np.array([0])
        selected_content = selected_content.replace("\n\n\n\nOne-sentence summary:", "")
        prefix, completion = create_prefix_completion(selected_content, summary[selected_idx])
        tokenized_summary = [self.tokenizer.eos_token_id] + self.tokenizer.encode(summary[selected_idx]) if 'bart' in self.tokenizer_model else self.tokenizer.encode(summary[selected_idx])
        tokenized_prefix = [self.tokenizer.eos_token_id] + self.tokenizer.encode(prefix) if 'bart' in self.tokenizer_model else self.tokenizer.encode(prefix)
        tokenized_completion = [self.tokenizer.eos_token_id] + self.tokenizer.encode(completion) if 'bart' in self.tokenizer_model else self.tokenizer.encode(completion)
        loss_mask = np.array([0 for _ in range(len(tokenized_prefix))] + [1 for _ in range(len(tokenized_completion) - len(tokenized_prefix))])

        if self.tokenized_info:
            # prefix_info: 'input_ids', 'attention_mask' (all 1)
            prefix_info = self.tokenizer(selected_content, return_tensors='pt')
            # completion_info: 'input_ids', 'attention_mask'
            completion_info = self.tokenizer(summary[selected_idx], return_tensors='pt')
            # reversed_prefix_sentence_info: 'input_ids', 'attention_mask'
            content_sentences = split_paragraphs(selected_content, mode='sentence')
            reversed_prefix_sentence_info = self.tokenizer(list(reversed([s for s in content_sentences if len(s.strip()) > 0])), return_tensors='pt', padding=True)
        else:
            prefix_info, completion_info, reversed_prefix_sentence_info = None, None, None

        example = {'prefix': tokenized_completion, # you actually want to run on all of the completion, and then mask out the tokenized_prefix sometimes
                   'labels': label, 
                   'summary': tokenized_summary, 
                   'loss_mask': loss_mask,
                   'prefix_info': prefix_info, 
                   'completion_info': completion_info,
                   'reversed_prefix_sentence_info': reversed_prefix_sentence_info,
                  }

        return example, completion