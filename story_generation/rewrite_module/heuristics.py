import argparse
import csv
from enum import auto
import os
from copy import deepcopy
import pickle
from collections import defaultdict
import multiprocessing as mp
import random
import string

import torch
import Levenshtein
import numpy as np
from transformers import AutoTokenizer
import openai
from scipy.special import softmax

from story_generation.edit_module.entity import *
from story_generation.common.util import *
from story_generation.common.data.data_util import add_data_args, load_dataset
from story_generation.common.summarizer.summarizer_util import add_summarizer_args, load_summarizer
from story_generation.common.summarizer.models.gpt3_summarizer import GPT3_SEP, GPT3_END
from story_generation.common.controller.controller_util import add_controller_args, load_controller
from story_generation.common.controller.loaders.alignment_loader import create_prefix_completion
from story_generation.common.data.split_paragraphs import *

def detect_first_second_person(text):
    text = text.split('"')
    for i in range(0, len(text), 2): # all the sections that are outside of quotations
        if 'I ' in text[i] or "I'" in text[i] or 'you ' in text[i].lower() or "you'" in text[i].lower() or " we " in text[i].lower() or "\nwe " in text[i].lower() or "we'" in text[i].lower():
            return True
    return False


def calculate_repetition_length_penalty(generation, prompt_sentences, levenshtein_repetition_threshold=0.8, max_length=None, tokenizer=None, is_outline=False):
    if len(generation.strip()) == 0:
        return 10
    if max_length is not None:
        if len(tokenizer.encode(generation)) > max_length:
            return 10
    if any([s.lower() in generation.lower() for s in ['\nRelevant', '\nContext', '\nComment', 'Summar', '\nSupporting', '\nEvidence', '\nStages', '\nText', '\nAssum', '\n1.', '\n1)', '\nRelationship', '\nMain Character', '\nCharacter', '\nConflict:', '\nPlot', 'TBA', 'POV', 'protagonist', '\nEdit ', '\nPremise', 'Suspense', 'www', '[', ']', 'copyright', 'chapter', '\nNote', 'Full Text', 'narrat', '\n(', 'All rights reserved', 'story', '(1)', 'passage', '\nRundown', 'playdown', 'episode', 'plot device', 'java', '\nQuestion', '\nDiscuss']]): # it's repeating parts of the prompt/reverting to analysis
        return 10
    generation_paragraphs = split_paragraphs(generation, mode='newline')
    for paragraph in generation_paragraphs:
        if len(paragraph.strip()) == 0:
            continue
        if ':' in ' '.join(paragraph.strip().split()[:10]) or paragraph.strip().endswith(':'): # there's a colon in the first few words, so it's probably a section header for some fake analysis, or ends with a colon
            return 10
    penalty = 0
    for p in prompt_sentences:
        split = p.lower().split(' ')
        for i in range(6, len(split)):
            if ' '.join(split[i-5:i]) in generation.lower(): # somewhat penalize repeated strings of 5 words or more for each prompt sentence
                penalty += 0.3
                # break
    split = generation.lower().split(' ')
    for i in range(6, len(split)):
        if ' '.join(split[i-5:i]) in ' '.join(split[i:]): # penalize repetition within the generation itself
            penalty += 0.3
            # break
    mildly_bad_strings = ['\n\n\n\n\n', 'story', 'stories', 'passage', 'perspective', 'point of view', 'summar', 'paragraph', 'sentence', 'example', 'analy', 'section', 'character', 'review', 'readers', '(', ')', 'blog', 'website', 'comment']
    if not is_outline:
        mildly_bad_strings += ['1.', '2.', '3.', '4.', '5.']
    num_mildly_bad_strings = sum([1 for s in mildly_bad_strings if s in generation.lower()])
    if num_mildly_bad_strings > 0:
        penalty += num_mildly_bad_strings # discourage multiple of these strings appearing, since it's likely that this is resulting from GPT3 generating story analysis
    generation_sentences = split_paragraphs(generation, mode='sentence')
    for g in generation_sentences:
        for p in prompt_sentences:
            if Levenshtein.ratio(g, p) > levenshtein_repetition_threshold:
                penalty += 1
    return penalty