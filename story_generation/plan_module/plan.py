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
import logging

import torch
import Levenshtein
import numpy as np
from transformers import AutoTokenizer
import openai
from scipy.special import softmax

from story_generation.edit_module.entity import *
from story_generation.rewrite_module.heuristics import *
from story_generation.common.util import *
from story_generation.common.data.data_util import add_data_args, load_dataset
from story_generation.common.summarizer.summarizer_util import add_summarizer_args, load_summarizer
from story_generation.common.summarizer.models.gpt3_summarizer import GPT3_SEP, GPT3_END
from story_generation.common.controller.controller_util import add_controller_args, load_controller
from story_generation.common.controller.loaders.alignment_loader import create_prefix_completion
from story_generation.common.data.split_paragraphs import *

def generate_initial_entity_strings(premise, setting, instruct_model, num_entities=3, max_description_length=48, model_string='text-davinci-002'):
    # TODO figure out alternative stopping criterion for generating initial characters?
    initial_characters_prompt = "Premise: " + premise.strip() + '\n\n' + 'Setting: ' + setting.strip() + '\n\nList the names and details of all major characters.'
    name_bias_words = ['protagonist', 'Protagonist', 'PROTAGONIST', 'unnamed', 'Unnamed', 'UNNAMED', 'unknown', 'Unknown', 'UNKNOWN', 'None', 'none', 'None', 'Mr', 'Ms', 'Mrs', 'Dr', 'TBA', 'TBD', 'N/A'] # technically no ' can filter out some reasonable names, but it's not a big deal and prevents some bad cases
    banned_name_words = name_bias_words + ['\'', '_', '\n', '"', '#', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'redacted', 'mother', 'father', 'gram', 'grand', 'name', 'appearance', 'occupation', 'age', 'gender', 'sex', 'role', 'profession', 'job', 'friend'] + list(string.punctuation) # sometimes it'll find weird ascii chars to replace these if they're banned via logit bias
    name_logit_bias = get_repetition_logit_bias(instruct_model.tokenizer, initial_characters_prompt + ' ' + ' '.join(name_bias_words), bias=-5, bias_common_tokens=True)
    name_logit_bias[198] = -5 # also penalize newline, although we want it eventually eventually

    character_strings = {}
    characters_prompt = initial_characters_prompt
    for i in range(num_entities):
        characters_prompt += '\n\n' + str(i+1) +'.\n\nFull Name:'
        for _ in range(2):
            name_continuations = instruct_model([characters_prompt], modify_prompt=False, top_p=1, temperature=1.2, logit_bias=name_logit_bias, stop=['\n', '(', ':'], num_completions=10, generation_max_length=10, model_string=model_string)
            filtered_name_continuations = []
            for name in name_continuations:
                name_is_good = True
                for word in name.strip().split():
                    if word.strip(string.punctuation) not in characters_prompt and sum([1 for n in name_continuations if word in n]) >= 2: # >=2 because it's in the name itself and at least 1 other
                        name_is_good = False
                        logging.log(23, 'bad name word ' + word + ' in ' + name)
                        for tok in instruct_model.tokenizer.encode(word) + instruct_model.tokenizer.encode(' ' + word):
                            name_logit_bias[tok] = -100
                if not name_is_good:
                    continue
                if not any([key.strip() in name.strip() or name.strip() in key.strip() for key in character_strings]) and len(name.strip()) > 0 and all([piece.strip()[0].isupper() for piece in name.strip().split()]) and all([word.lower() not in name.lower() for word in banned_name_words+name_bias_words]): # check that names are capitalized to filter out some bad cases
                    if not any([word.strip('"') not in initial_characters_prompt and word.lower() in initial_characters_prompt.lower() for word in name.strip().split()]) and sum([1 for letter in name if letter.isupper()]) == len(name.strip().split()): # don't allow cases where it dodged our checks by changing case
                        filtered_name_continuations.append(name)
            if len(filtered_name_continuations) > 0:
                break
        if len(filtered_name_continuations) == 0:
            if len(character_strings) > 0: # just settle for fewer characters
                break
            else:
                raise ValueError
        filtered_name_continuations = sorted(filtered_name_continuations, key=lambda x: abs(2 - len(x.strip().split()))) # ideally want the full name, not just the first word, and want roughly 2 words
        selected_name = filtered_name_continuations[0].strip()
        name_logit_bias = get_repetition_logit_bias(instruct_model.tokenizer, selected_name.strip().split()[0], bias=-100, bias_common_tokens=True, existing_logit_bias=name_logit_bias)
        banned_name_words.append(selected_name.strip().split()[0])
        characters_prompt += ' ' + selected_name + '\n\nCharacter Portrait: ' + selected_name.strip() + ' is'
        found_acceptable_description = False
        logging.log(21, 'CHARACTERS PROMPT', characters_prompt)
        for j in range(5):
            description_logit_bias = get_repetition_logit_bias(instruct_model.tokenizer, initial_characters_prompt + ' ' + ' '.join(name_bias_words), bias=-2**(j+1), bias_common_tokens=False)
            name_tokens = set(sum([instruct_model.tokenizer.encode(ent) + instruct_model.tokenizer.encode(' ' + ent) for ent in character_strings.keys()], []))
            for tok in name_tokens:
                if tok in description_logit_bias:
                    del description_logit_bias[tok]
            descriptions = instruct_model([characters_prompt], modify_prompt=False, stop='\n', logit_bias=description_logit_bias, num_completions=10, generation_max_length=max_description_length, cut_sentence=True, model_string=model_string)
            logging.log(21, 'DESCRIPTIONS', descriptions)
            descriptions = [d for d in descriptions if len(d.strip()) > 0 and len(instruct_model.tokenizer.encode(d)) < max_description_length] # not empty, and terminated naturally rather than due to max length
            descriptions = sorted(descriptions, key=lambda d: calculate_repetition_length_penalty(d, [characters_prompt]))
            if len(descriptions) > 0 and calculate_repetition_length_penalty(descriptions[0], [characters_prompt]) < 1:
                found_acceptable_description = True
                break
        if not found_acceptable_description:
            logging.warning('Warning: no acceptable description found for character ' + selected_name)
            assert False
        description = descriptions[0]
        characters_prompt += description
        character_strings[selected_name.strip()] = Entity(selected_name.strip(), description=selected_name.strip() + ' is' + description, is_character=True)
    infer_attributes_string = premise.strip() + '\n\n' + setting.strip() + '\n\n' + '\n\n'.join([ent.description for ent in character_strings.values()])
    return characters_prompt[len(initial_characters_prompt):].strip(), character_strings, infer_attributes_string


def generate_outline(premise, setting, characters, character_strings, instruct_model, generation_max_length, max_sections=5, fixed_outline_length=-1, outline_levels=1, model_string='text-davinci-002'):
    premise_setting_chars = "Premise: " + premise.strip() + '\n\n' + 'Setting: ' + setting.strip() + '\n\n' + 'Characters: ' + characters.strip()

    if fixed_outline_length > 0:
        outline_prompt = premise_setting_chars + '\n\n\n\nOutline the ' + str(fixed_outline_length) + ' main plot points of the story.\n\n1.'
    else:
        outline_prompt = premise_setting_chars + '\n\n\n\nOutline the main plot points of the story.\n\n1.'
    found_acceptable_outline = False
    for i in range(5):
        # bias against repeating the tokens in the prompt, except for the character names themselves
        outline_logit_bias = get_repetition_logit_bias(instruct_model.tokenizer, outline_prompt, -2**(i+1))
        name_tokens = set(sum([instruct_model.tokenizer.encode(ent) + instruct_model.tokenizer.encode(' ' + ent) for ent in character_strings.keys()], []))
        for tok in name_tokens:
            if tok in outline_logit_bias:
                del outline_logit_bias[tok]
        outlines = instruct_model([outline_prompt], logit_bias=outline_logit_bias, generation_max_length=generation_max_length, num_completions=5, model_string=model_string)
        for outline in outlines:
            if fixed_outline_length > 0:
                if str(fixed_outline_length) + '.' not in outline or str(fixed_outline_length+1) + '.' in outline: # looking for exactly this length
                    continue
            if len(split_list('1.' + outline)) < 3: # failure
                continue
            if '2.' not in outline or '3.' not in outline: # properly formatted list and contains at least 3 items
                continue
            if str(max_sections) + '.' in outline: # number of sections in outline exceeds maximum
                continue
            if calculate_repetition_length_penalty(outline, [setting, characters], is_outline=True) > 0: # it's fine if some of the premise is repeated e.g. in the early parts
                continue
            if len(instruct_model.tokenizer.encode(outline)) < generation_max_length: # ideally, terminate because the outline is done, not because it was too long
                found_acceptable_outline = True
                break
        if found_acceptable_outline:
            break
    if not found_acceptable_outline:
        logging.warning('Warning: didn\'t find acceptable outline')
        raise ValueError
    outline = ('1.' + outline).strip()
    logging.log(23, outline)
    if outline_levels > 1:
        all_detailed_outlines = []
        assert outline_levels == 2 # in principle could support more
        for outline_idx, outline_piece in enumerate(split_list(outline)):
            found_acceptable_outline = False
            for i in range(5):
                detailed_outline_logit_bias = get_repetition_logit_bias(instruct_model.tokenizer, outline_prompt + ' ' + ' '.join([op for op in split_list(outline)]), -2**(i+1))
                name_tokens = set(sum([instruct_model.tokenizer.encode(ent) + instruct_model.tokenizer.encode(' ' + ent) for ent in character_strings.keys()], []))
                for tok in name_tokens:
                    if tok in outline_logit_bias:
                        del outline_logit_bias[tok]
                detailed_outlines = instruct_model([premise_setting_chars + '\n\nOutline:\n\n' + '\n\n'.join([op for op in split_list(outline)[:outline_idx]]) + '\n\nList the minor events in the next part of the story, in which ' + outline_piece.strip() + '\n\n1.'], logit_bias=detailed_outline_logit_bias, generation_max_length=generation_max_length, num_completions=5, model_string=model_string)
                for detailed_outline in detailed_outlines:
                    if fixed_outline_length > 0:
                        if str(fixed_outline_length) + '.' not in detailed_outline or str(fixed_outline_length+1) + '.' in detailed_outline: # looking for exactly this length
                            continue
                    if len(split_list('1.' + detailed_outline)) < 3: # failure
                        continue
                    if '2.' not in detailed_outline or '3.' not in detailed_outline: # properly formatted list and contains at least 3 items
                        continue
                    if str(max_sections) + '.' in detailed_outline: # number of sections in outline exceeds maximum
                        continue
                    if calculate_repetition_length_penalty(detailed_outline, [setting, characters, outline], is_outline=True) > 0: # it's fine if some of the premise is repeated e.g. in the early parts
                        continue
                    if len(instruct_model.tokenizer.encode(detailed_outline)) < generation_max_length: # ideally, terminate because the outline is done, not because it was too long
                        found_acceptable_outline = True
                        break
                if found_acceptable_outline:
                    break
            if not found_acceptable_outline:
                logging.log(23, 'Warning: didn\'t find acceptable outline')
                raise ValueError
            all_detailed_outlines.append('1.' + detailed_outline)
        outline = (outline, all_detailed_outlines)
    return outline


def load_plan_info(plan_file):
    with open(plan_file, 'rb') as f:
        save_info = pickle.load(f)
    return save_info


def generate_plan_info(args, instruct_model, include_outline=True, model_string='text-davinci-002'):
    while True:
        try:
            if args.premise is None:
                premise_prompt = "Write a premise for a short story."
                max_premise_tokens = 128
                premise = (instruct_model([premise_prompt], top_p=1, temperature=1.2, modify_prompt=False, generation_max_length=max_premise_tokens, model_string=model_string)[0]) # more diversity with premises with higher temp
                if len(instruct_model.tokenizer.encode(premise)) == max_premise_tokens: # likely we got cutoff instead of ending naturally
                    logging.warning('premise too long, retrying')
                    raise ValueError
                premise = premise.strip()
            else:
                premise = args.premise.strip()

            logging.log(25, 'Premise: ' + premise)

            for i in range(10): # avoid resampling good premises for fairness
                try:
                    setting_prompt = "Premise: " + premise.strip() + '\n\nDescribe the setting of the story.\n\nThe story is set in'
                    settings = []
                    for i in range(5):
                        banned_setting_words = ['unknown', 'unnamed', 'unspecified', 'Unknown', 'Unnamed', 'Unspecified']
                        setting_logit_bias = get_repetition_logit_bias(instruct_model.tokenizer, setting_prompt, -2**(i+1))
                        settings = instruct_model([setting_prompt], num_completions=10, modify_prompt=False, logit_bias=setting_logit_bias, generation_max_length=32, cut_sentence=True, model_string=model_string)
                        settings = [split_paragraphs(s, mode='sentence')[0] for s in settings]
                        settings = [s.strip() for s in settings if calculate_repetition_length_penalty(s, [premise]) == 0 and not any([w in s.lower() for w in banned_setting_words])]
                        settings = ['The story is set in ' + s for s in settings]
                        if len(settings) > 0:
                            break
                    setting = settings[0]

                    logging.log(25, 'Setting: ' + setting)

                    characters, character_strings, infer_attributes_string = generate_initial_entity_strings(premise, setting, instruct_model, max_description_length=args.entity_description_max_length, model_string=model_string)

                    logging.log(25, 'Characters: ' + str(characters))

                    for entity in character_strings.values():
                        logging.log(23, entity)

                    if not include_outline:
                        outline = None
                        break
                    outline_max_tokens = 128
                    outline = generate_outline(premise, setting, characters, character_strings, instruct_model, outline_max_tokens, fixed_outline_length=args.fixed_outline_length, outline_levels=args.outline_levels)

                    # assume gpt3 was smart enough to number them when prompted
                    if type(outline) == tuple:
                        outline_sections = sum([split_list(op) for op in outline[1]], [])
                    else:
                        outline_sections = split_list(outline)

                    logging.log(25, 'Outline: ' + str(outline))

                    # do the attribute inference after outlines are generated, since it can be expensive
                    if not args.no_attributes and not args.no_editor and not args.no_planner:
                        for entity in character_strings.values():
                            entity.infer_attributes(infer_attributes_string, instruct_model, other_names=[name for name in character_strings.keys() if name != entity.name])
                        complete_mutual_relations(character_strings, instruct_model)
                    break
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    logging.log(23, 'Plan generation failed: ' + str(e))
            if i == 9:
                logging.warning('WARNING: Could not generate a valid setup after 10 attempts.')
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            logging.warning('Exception ' + str(e))
            continue
    save_info = {'premise': premise,
                'setting': setting,
                'characters': characters,
                'character_strings': character_strings,
                'outline': outline,
                'outline_sections': outline_sections,
                'infer_attributes_string': infer_attributes_string}
    return save_info


def infer_initial_attributes_from_plan(save_info, instruct_model):
    character_strings = save_info['character_strings']
    infer_attributes_string = save_info['infer_attributes_string']
    made_changes = False
    for entity in character_strings.values():
        if len(entity.attributes) == 0 and entity.is_character: # unlikely that we inferred nothing from an initial setup passage
            made_changes = True
            entity.infer_attributes(infer_attributes_string, instruct_model, other_names=[name for name in character_strings.keys() if name != entity.name])
    if made_changes:
        complete_mutual_relations(character_strings, instruct_model)