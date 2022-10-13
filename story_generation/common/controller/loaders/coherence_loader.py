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

from story_generation.common.data.split_paragraphs import split_paragraphs, group_chunks
from story_generation.common.data.tree_util import START_OF_STORY, MIDDLE_OF_STORY

class CoherenceSplitLoader(torch.utils.data.IterableDataset):
    def __init__(self, contents, summaries, tokenizer_model, append_mask_token=False, time_label_decay=1, **kwargs):
        super(CoherenceSplitLoader).__init__()
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
        self.use_special_tokens = kwargs['use_special_tokens'] if 'use_special_tokens' in kwargs else False
        self.negative_categories = kwargs['negative_categories'] if 'negative_categories' in kwargs else ['other', 'repeat', 'shuffle']
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
            try:
                summary = self.summaries[self.pos]
                tokenized_summary = self.tokenizer.encode(summary)
                base_content = self.contents[self.pos]

                # segment into sentences
                sentences = split_paragraphs(base_content, mode='sentence')
                if len(sentences) < 2:
                    self.pos += increment
                    continue
                sentences = group_chunks(sentences, max_chunk_length=200) # so actually paragraphs
                if self.use_special_tokens:
                    sentences = [START_OF_STORY] + sentences
                    if random.random() < 0.5: # chance to cutoff mid-story
                        try:
                            pre_cutoff = random.randint(2, len(sentences) - 1) # if omitting previous details, omit something other than start token
                        except:
                            self.pos += increment
                            continue
                        sentences = [MIDDLE_OF_STORY] + sentences[pre_cutoff:]
                # cutoff at some sentence
                try:
                    cutoff = random.randint(1 if self.use_special_tokens else 0, len(sentences)-1)
                except:
                    self.pos += increment
                    continue
                prefix = ' '.join([s.strip() for s in sentences[:cutoff]])
                if len(prefix.strip()) == 0:
                    self.pos += increment
                    continue
                tokenized_prefix = [self.tokenizer.eos_token_id] + self.tokenizer.encode(prefix) if 'bart' in self.tokenizer_model else self.tokenizer.encode(prefix)
                
                # select true, repetition, shuffled sentence, random other story
                possible_modes = ['true']
                if 'other' in self.negative_categories:
                    possible_modes.append('other')
                if cutoff > 0 and 'repeat' in self.negative_categories: # can't repeat if don't have anything to repeat yet
                    possible_modes.append('repeat')
                if cutoff < len(sentences) - 1 and 'shuffle' in self.negative_categories: # no shuffle if only 1 sentence left
                    possible_modes.append('shuffle')
                
                if self.generate_negatives:
                    completions = set()
                    all_examples = []
                    true_example, true_completion = self.create_example('true', sentences, cutoff, prefix, tokenized_prefix, tokenized_summary)
                    all_examples.append(true_example)
                    completions.add(true_completion)
                    for _ in range(self.num_negatives):
                        while True:
                            mode = random.choice(possible_modes)
                            if mode == 'true':
                                continue
                            neg_example, neg_completion = self.create_example(mode, sentences, cutoff, prefix, tokenized_prefix, tokenized_summary)
                            if neg_completion not in completions:
                                all_examples.append(neg_example)
                                completions.add(neg_completion)
                                break
                else:
                    mode = random.choice(possible_modes)
                    example, _ = self.create_example(mode, sentences, cutoff, prefix, tokenized_prefix, tokenized_summary)
                    all_examples = example

                valid = True
                self.pos += increment
            except:
                self.pos += increment
                continue
        return all_examples
    
    def create_example(self, mode, sentences, cutoff, prefix, tokenized_prefix, tokenized_summary):
        if mode == 'true':
            separate_completion = sentences[cutoff]
            # completion = ' '.join([s.strip() for s in sentences[:cutoff+1]])
            completion = prefix.strip() + ' ' + separate_completion
            label = np.array([1])
        # create repetition example
        elif mode == 'repeat':
            separate_completion = random.choice(sentences[:cutoff]).strip() # random already used sentence
            completion = prefix.strip() + ' ' + separate_completion
            label = np.array([0])
        # create shuffled sentence example
        elif mode == 'shuffle':
            separate_completion = random.choice(sentences[cutoff+1:]).strip() # random out of order sentence
            completion = prefix.strip() + ' ' + separate_completion
            label = np.array([0])
        # create random other story example
        elif mode == 'other':
            other_content_sentences = []
            while len(other_content_sentences) == 0:
                other_content = self.contents[random.randint(0, len(self.contents)-1)]
                other_content_sentences = split_paragraphs(other_content, mode='sentence')
                other_content_sentences = group_chunks(other_content_sentences, max_chunk_length=200) # so actually paragraphs
            separate_completion = random.choice(other_content_sentences).strip()
            completion = prefix.strip() + ' ' + separate_completion
            label = np.array([0])
        # print('MODE', mode)
        # print('PREFIX', prefix)
        # print('SEPARATE COMPLETION', separate_completion)
        # import pdb; pdb.set_trace()
        tokenized_completion = [self.tokenizer.eos_token_id] + self.tokenizer.encode(completion) if 'bart' in self.tokenizer_model else self.tokenizer.encode(completion)
        loss_mask = np.array([0 for _ in range(len(tokenized_prefix))] + [1 for _ in range(len(tokenized_completion) - len(tokenized_prefix))])

        if self.tokenized_info:
            # prefix_info: 'input_ids', 'attention_mask' (all 1)
            prefix_info = self.tokenizer(prefix, return_tensors='pt')
            # completion_info: 'input_ids', 'attention_mask'
            completion_info = self.tokenizer(separate_completion, return_tensors='pt')
            # reversed_prefix_sentence_info: 'input_ids', 'attention_mask'
            reversed_prefix_sentence_info = self.tokenizer(list(reversed([s for s in sentences[:cutoff] if len(s.strip()) > 0])), return_tensors='pt', padding=True)
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