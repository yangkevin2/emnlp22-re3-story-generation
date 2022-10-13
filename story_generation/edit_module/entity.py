from audioop import reverse
from collections import Counter, defaultdict
import multiprocessing as mp
from functools import partial
import os
import time
import string
import pathlib
import csv
import random
from unicodedata import name
import math
import logging

import torch
import openai
from scipy.special import softmax
import numpy as np

from story_generation.common.util import *
from story_generation.common.data.split_paragraphs import split_paragraphs


ENTITY_MODEL_STRING = 'text-curie-001'
STRONGER_ENTITY_MODEL_STRING = 'text-davinci-002'


example_library = None


def get_example_library():
    global example_library
    if example_library is None:
        example_lines = []
        with open(os.path.join(pathlib.Path(__file__).parent.resolve(), 'example_library.csv'), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                example_lines.append(row)
        example_library = {
                            'sentences': [example['text'] for example in example_lines], 
                            'names': [example['name'] for example in example_lines],
                            'keys': [example['key'] for example in example_lines],
                            'values': [example['value'] for example in example_lines],
                            'encodings': sentence_encode([example['text'] for example in example_lines]),
                          }
    return example_library


def construct_infobox_prompt(current_sentence, current_name, other_names, num_examples=5, random_order=False):
    instruction = 'Extract attributes from the given context using the format Attribute: Value.\n----'
    example_library = get_example_library()
    current_encoding = sentence_encode([current_sentence])
    scores = (current_encoding * example_library['encodings']).sum(axis=1)
    scores_indices = list(range(len(scores)))
    best_example_indices = sorted(scores_indices, key=lambda i: scores[i], reverse=True)
    best_example_indices = [i for i in best_example_indices if all([tok not in example_library['sentences'][i] for tok in current_name.split() + sum([other_name.split() for other_name in other_names], [])])] # don't get weird examples from lucky matching names
    best_example_indices = [i for i in best_example_indices if all([tok not in current_name and tok not in current_sentence for tok in example_library['names'][i].split()])]
    best_example_indices = best_example_indices[:num_examples]
    best_example_indices = reversed(best_example_indices) # put the best ones last
    if random_order:
        random.shuffle(best_example_indices)
    for i in best_example_indices:
        name = example_library['names'][i]
        instruction = instruction + '\nContext (' + name + '): ' + example_library['sentences'][i]
        keys = [key.strip() for key in example_library['keys'][i].split(',') if len(key.strip()) > 0]
        values = [value.strip() for value in example_library['values'][i].split(',') if len(value.strip()) > 0]
        assert len(keys) == len(values)
        for key, value in zip(keys, values):
            if key.endswith('\'s'):
                instruction = instruction + '\n' + name + ' is ' + key + ' ' + value # John is Jane's X
            else:
                instruction = instruction + '\n' + name + '\'s ' + key + ' is ' + value # John's sister is X
        instruction = instruction + '\n----'
    return instruction + '\nContext (' + current_name + '): ' + current_sentence + '\n' + current_name


def resample_name(name, context, instruct_model):
    name_tokens = name.strip().split()
    banned_tokens = [tok for tok in name_tokens if tok not in context]
    banned_tokens = banned_tokens + [' ' + tok for tok in banned_tokens]
    logit_bias = {}
    for name_token in banned_tokens:
        for tok in instruct_model.tokenizer.encode(name_token):
            logit_bias[tok] = -100
    for _ in range(10):
        new_name = instruct_model(["Generate a name with the same gender similar to " + name + ".\n\nName:"], logit_bias=logit_bias, stop=[',', '.'], top_p=1, generation_max_length=10, model_string=ENTITY_MODEL_STRING)[0].strip()
        if len(new_name) < 10: # ideally, it should terminate due to a stop, and not max length
            break 
    return new_name


def split_key_entry(infobox_entry, entity_name):
    try:
        if infobox_entry.startswith(entity_name + '\'s'):
            new_key, new_entry = tuple(infobox_entry.split(' is '))
            new_key = new_key[len(entity_name + '\'s'):]
        else:
            assert infobox_entry.startswith(entity_name + ' is')
            split_entry = infobox_entry.split(' is ')
            assert len(split_entry) == 2
            new_key_entry = split_entry[1]
            split_new_key_entry = new_key_entry.split('\'') # James's sister
            assert len(split_new_key_entry) == 2
            new_key = split_new_key_entry[0].strip() + '\'s'
            new_entry = split_new_key_entry[1][1:].strip() # remove the training s from 's constructions, or a space if not there
    except:
        return None, None
    return new_key.strip(), new_entry.strip()


def qa_entailment_best_answer(question, context, num_beams=1):
    new_qa_entries_passage, new_qa_scores_passage = score_qa(question, context, num_beams=num_beams)
    qa_passage_dict = defaultdict(lambda: 0)
    for answer, score in zip(new_qa_entries_passage, new_qa_scores_passage):
        qa_passage_dict[question + ' ' + answer] += score
    qa_passage_groups = get_entailment_groups(qa_passage_dict)
    new_qa_entry_passage = sorted(list(qa_passage_groups.keys()), key=lambda x: qa_passage_groups[x], reverse=True)[0]
    new_qa_score_passage = qa_passage_groups[new_qa_entry_passage]
    return new_qa_entry_passage, new_qa_score_passage


def complete_mutual_relations(entities, instruct_model, return_contradiction_prob=False):
    contradictions = defaultdict(lambda: {})
    contradiction_prob = 0
    entity_key_pairs = []
    for entity in entities.values():
        for key in entity.attributes.keys():
            entity_key_pairs.append((entity.name, key, entity.attributes[key]['entailment']))
    # go through (entity, key) pairs in order of entailment
    entity_key_pairs = sorted(entity_key_pairs, key=lambda x: x[2], reverse=True)
    for entity_name, key, _ in entity_key_pairs:
        entity = entities[entity_name]
        if entity.attributes[key]['newly_entered']:
            # match entities in the newly entered attributes; if you have exactly 1 other entity then:
            key_matched_entities, _, _ = deduplicate_match_entities(detect_entities(key), [n for n in entities.keys() if n != entity.name])
            value_matched_entities, _, _ = deduplicate_match_entities(detect_entities(entity.attributes[key]['entry']), [n for n in entities.keys() if n != entity.name])
            if len(key_matched_entities) + len(value_matched_entities) == 1:
                # if other entity is key + 's, then flip the value and key while stripping 's, also flipping the relation. 
                other_name = list(key_matched_entities)[0] if len(key_matched_entities) == 1 else list(value_matched_entities)[0]
                if len(key_matched_entities) == 1:
                    if not key.endswith('\'s'): # e.g. Sarah's
                        continue
                    if len(key.strip().split()) > 2: # might miss some longer names, but this will catch most bad cases where it's not actually a relation
                        continue
                    self_is_others_relation = entity.attributes[key]['entry'].strip()
                    if len(self_is_others_relation.strip().split()) > 1:
                        # if the value is multiple words, then try to parse and keep a single NN span only. if that still fails, then just skip this one?, with log
                        spans = pos_tag(self_is_others_relation)
                        self_is_others_relation = ' '.join([s.text for s in spans if s.tag in ['NN', 'NNS', 'NNP', 'NNPS']])
                        if len(self_is_others_relation.strip().split()) > 1: # multiple nouns in the entry
                            logging.log(23, 'FAILED TO PARSE MUTUAL RELATION: ' + entity.name + ' ' + key + ' ' + str(entity.attributes[key]))
                            continue
                    prompt = entity.name + ' is ' + other_name + '\'s ' + self_is_others_relation + '. ' + other_name + ' is ' + entity.name + '\'s'
                    try:
                        other_is_selfs_relation = Counter([x.strip() for x in instruct_model([prompt], stop=['\n', '.', ','], num_completions=10, model_string=STRONGER_ENTITY_MODEL_STRING) if len(x.strip()) > 0]).most_common()[0][0]
                    except:
                        logging.log(23, 'FAILED TO GET OTHER RELATION: ' +  entity.name + ' ' + key + ' ' + str(entity.attributes[key]) + ' ' + prompt)
                        continue
                else:
                    # if other entity in the value + 's name, then flip it to key + 's and change the current key to the value, also flipping the relation. 
                    if not key.endswith('\'s name'): # e.g., spouse's name
                        continue
                    other_is_selfs_relation = key.replace('\'s name', '').strip()
                    prompt = other_name + ' is ' + entity.name + '\'s ' + other_is_selfs_relation + '. ' + entity.name + ' is ' + other_name + '\'s'
                    try:
                        self_is_others_relation = Counter([x.strip() for x in instruct_model([prompt], stop=['\n', '.', ','], num_completions=10, model_string=STRONGER_ENTITY_MODEL_STRING) if len(x.strip()) > 0]).most_common()[0][0]
                    except:
                        logging.log(23, 'FAILED TO GET OTHER RELATION:' + ' ' + entity.name + ' ' + key + ' ' + str(entity.attributes[key]) + ' ' + prompt)
                        continue

                logging.log(23, 'CHECKING MUTUAL RELATION FOR CONTRADICTIONS: ' + ' ' + entity.name + ' ' + key + ' ' + str(entity.attributes[key]))
                
                if len(key_matched_entities) == 1:
                    if other_is_selfs_relation + '\'s name' in entity.attributes and not entity.attributes[other_is_selfs_relation + '\'s name']['newly_entered']:
                        existing_entailment_input = entity.create_entailment_input(other_is_selfs_relation + '\'s name', entity.name, entity.attributes[other_is_selfs_relation + '\'s name']['entry'])
                        new_entailment_input = entity.create_entailment_input(other_is_selfs_relation + '\'s name', entity.name, other_name)
                        if not consistent_equals(existing_entailment_input, new_entailment_input):
                            logging.log(23, 'POTENTIAL MUTUAL RELATION CONTRADICTION')
                            logging.log(23, 'PREEXISTING' + ' ' + entity.name + ' ' + other_is_selfs_relation + '\'s name' + ' ' + str(entity.attributes[other_is_selfs_relation + '\'s name']))
                            logging.log(23, 'NEW' + ' ' + entity.name + ' ' + key + ' ' + str(entity.attributes[key]))
                            contradictions[entity.name][other_is_selfs_relation + '\'s name'] = (entity.attributes[other_is_selfs_relation + '\'s name'], {'key': key, 'entry': entity.attributes[key]})
                            if not return_contradiction_prob:
                                continue
                        contradiction_prob = max(contradiction_prob, math.exp(score_entailment(existing_entailment_input, new_entailment_input)[0][0, 0]))
                else:
                    if other_name + '\'s' in entity.attributes and not entity.attributes[other_name + '\'s']['newly_entered']:
                        existing_entailment_input = entity.create_entailment_input(other_name + '\'s', entity.name, entity.attributes[other_name + '\'s']['entry'])
                        new_entailment_input = entity.create_entailment_input(other_name + '\'s', entity.name, self_is_others_relation)
                        if not consistent_equals(existing_entailment_input, new_entailment_input):
                            logging.log(23, 'POTENTIAL MUTUAL RELATION CONTRADICTION')
                            logging.log(23, 'PREEXISTING' + ' ' + entity.name + ' ' + other_name + '\'s' + ' ' + str(entity.attributes[other_name + '\'s']))
                            logging.log(23, 'NEW' + ' ' + entity.name + ' ' + key + ' ' + str(entity.attributes[key]))
                            contradictions[entity.name][other_name + '\'s'] = (entity.attributes[other_name + '\'s'], {'key': key, 'entry': entity.attributes[key]})
                            if not return_contradiction_prob:
                                continue
                        contradiction_prob = max(contradiction_prob, math.exp(score_entailment(existing_entailment_input, new_entailment_input)[0][0, 0]))
                # check the corresponding relations for the other entity. 
                other_entity = entities[other_name]
                if self_is_others_relation + '\'s name' in other_entity.attributes and not other_entity.attributes[self_is_others_relation + '\'s name']['newly_entered']:
                    existing_entailment_input = other_entity.create_entailment_input(self_is_others_relation + '\'s name', other_name, other_entity.attributes[self_is_others_relation + '\'s name']['entry'])
                    new_entailment_input = other_entity.create_entailment_input(self_is_others_relation + '\'s name', other_name, entity.name)
                    if not consistent_equals(existing_entailment_input, new_entailment_input):
                        logging.log(23, 'POTENTIAL MUTUAL RELATION CONTRADICTION')
                        logging.log(23, 'PREEXISTING' + ' ' + other_name+ ' ' + self_is_others_relation + '\'s name' + ' ' + str(other_entity.attributes[self_is_others_relation + '\'s name']))
                        logging.log(23, 'NEW' + ' ' + entity.name + ' ' + key+ ' ' + str(entity.attributes[key]))
                        contradictions[other_name][self_is_others_relation + '\'s name'] = (other_entity.attributes[self_is_others_relation + '\'s name'], {'key': key, 'entry': entity.attributes[key]})
                        if not return_contradiction_prob:
                            continue
                    contradiction_prob = max(contradiction_prob, math.exp(score_entailment(existing_entailment_input, new_entailment_input)[0][0, 0]))
                if entity.name + '\'s' in other_entity.attributes and not other_entity.attributes[entity.name + '\'s']['newly_entered']:
                    existing_entailment_input = other_entity.create_entailment_input(entity.name + '\'s', other_name, other_entity.attributes[entity.name + '\'s']['entry'])
                    new_entailment_input = other_entity.create_entailment_input(entity.name + '\'s', other_name, other_is_selfs_relation)
                    if not consistent_equals(existing_entailment_input, new_entailment_input):
                        logging.log(23, 'POTENTIAL MUTUAL RELATION CONTRADICTION')
                        logging.log(23, 'PREEXISTING' + ' ' + other_name + ' ' + entity.name + '\'s' + ' ' + str(other_entity.attributes[entity.name + '\'s']))
                        logging.log(23, 'NEW' + ' ' + entity.name + ' ' + key + ' ' + str(entity.attributes[key]))
                        contradictions[other_name][entity.name + '\'s'] = (other_entity.attributes[entity.name + '\'s'], {'key': key, 'entry': entity.attributes[key]})
                        if not return_contradiction_prob:
                            continue
                    contradiction_prob = max(contradiction_prob, math.exp(score_entailment(existing_entailment_input, new_entailment_input)[0][0, 0]))
                if len(contradictions) > 0:
                    continue                  
                
                logging.log(23, 'COMPLETING MUTUAL RELATION:' + ' ' + entity.name + ' ' + key + ' ' +  str(entity.attributes[key]))
                if len(key_matched_entities) == 1:
                    _, change_status = entity.add_if_better(other_is_selfs_relation + '\'s name', {'text': entity.attributes[key]['text'],
                                                                                'entry': other_name, 
                                                                                'entailment': entity.attributes[key]['entailment'],
                                                                                'newly_entered': True}, detect_contradictions=False, return_contradiction_prob=return_contradiction_prob)
                    if change_status != 'none':
                        logging.log(23, 'NEW RELATION' + ' ' + change_status + ' ' + entity.name + ' ' + other_is_selfs_relation + '\'s name' + ' ' + str(entity.attributes[other_is_selfs_relation + '\'s name']))
                else:
                    _, change_status = entity.add_if_better(other_name + '\'s', {'text': entity.attributes[key]['text'],
                                                                'entry': self_is_others_relation,
                                                                'entailment': entity.attributes[key]['entailment'],
                                                                'newly_entered': True}, detect_contradictions=False, return_contradiction_prob=return_contradiction_prob)
                    if change_status != 'none':
                        logging.log(23, 'NEW RELATION' + ' ' + change_status + ' ' + entity.name + ' ' + other_name + '\'s' + ' ' + str(entity.attributes[other_name + '\'s']))
                _, change_status = other_entity.add_if_better(self_is_others_relation + '\'s name', {'text': entity.attributes[key]['text'],
                                                                                    'entry': entity.name, 
                                                                                    'entailment': entity.attributes[key]['entailment'],
                                                                                    'newly_entered': True}, detect_contradictions=False, return_contradiction_prob=return_contradiction_prob)
                if change_status != 'none':
                    logging.log(23, 'NEW RELATION' + ' ' + change_status + ' ' + other_name + ' ' + self_is_others_relation + '\'s name' + ' ' + str(other_entity.attributes[self_is_others_relation + '\'s name']))
                _, change_status = other_entity.add_if_better(entity.name + '\'s', {'text': entity.attributes[key]['text'],
                                                                    'entry': other_is_selfs_relation,
                                                                    'entailment': entity.attributes[key]['entailment'],
                                                                    'newly_entered': True}, detect_contradictions=False, return_contradiction_prob=return_contradiction_prob)
                if change_status != 'none':
                    logging.log(23, 'NEW RELATION' + ' ' + change_status + ' ' + other_name + ' ' + entity.name + '\'s' + ' ' + str(other_entity.attributes[entity.name + '\'s']))

    # change all newly entered attributes to false
    for entity in entities.values():
        for key in entity.attributes:
            entity.attributes[key]['newly_entered'] = False
    return entities, contradiction_prob if return_contradiction_prob else contradictions


class Entity:
    BANNED_ATTRIBUTES = ['personality', 'eye', 'hair'] # disproportionately hallucinated / problematic
    ENTAILMENT_THRESHOLD = 0.5
    FACT_ENTAILMENT_THRESHOLD = 0.3
    ENTAILMENT_RECHECK_THRESHOLD = 0.9
    CHARACTER_THRESHOLD = 0.5
    QA_SCORE_THRESHOLD = 0.5
    NO_ANSWER_WORDS = ['unknown', 'not ', 'unspecified', 'n/a', 'stated', 'mentioned', 'no answer', 'tba', 'tbd', 'never']
    def __init__(self, name, description=None, is_character=None, attributes=None):
        self.name = name
        self.description = description
        self.is_character = is_character
        self.attributes = attributes if attributes is not None else {}
    
    def __str__(self):
        formatted = self.name + ': ' + self.description + '\n' + 'Is Character: ' + str(self.is_character) + '\n' + 'Attributes: ' + str(self.attributes) + '\n'
        for attribute in self.attributes:
            formatted += attribute + ': ' + self.attributes[attribute]['entry'] + '\n'
        return formatted
    
    def reset_attributes(self):
        self.attributes = {}
    
    def create_entailment_input(self, key, name, value):
        if key.endswith('\'s'):
            return (name + ' is ' + key + ' ' + value).strip() # e.g. character is other character's sibling
        else:
            return (name + '\'s ' + key + ' is ' + value).strip()

    def get_referred_name(self, passage):
        return ' '.join([tok for tok in self.name.strip().split() if tok in passage])
    
    def resample_entry(self, info_entry, fact, gpt3_model, num_samples=3, split_entry=False):
        """
        Resample the entry based on the fact and the model.
        """
        # time.sleep(0.5)
        if split_entry:
            key, _ = split_key_entry(info_entry, self.name)
        else:
            key = info_entry
        if key is None:
            return info_entry
        prompt = self.create_entailment_input(key, self.name, '')
        candidate_entries = [entry for entry in gpt3_model([fact + '\n\n' + prompt], stop=['\n', '.', ','], num_completions=num_samples, top_p=1, temperature=0.8, model_string=ENTITY_MODEL_STRING) if len(entry.strip()) > 0 and entry.strip() != self.name]
        if len(candidate_entries) == 0:
            return None
        fact_entailment_scores, _ = score_entailment([fact for _ in range(len(candidate_entries))], 
                                                     [self.create_entailment_input(key, self.get_referred_name(fact), entry) for entry in candidate_entries])
        fact_entailment_probs = softmax(fact_entailment_scores, axis=1)
        candidate_entries = [candidate_entries[i] for i in range(len(candidate_entries)) if fact_entailment_probs[i, 2] > self.ENTAILMENT_THRESHOLD]
        if len(candidate_entries) == 0:
            return None
        entry_counter = Counter(candidate_entries)
        return prompt + entry_counter.most_common()[0][0]

    @torch.no_grad()
    def infer_description(self, passage, gpt3_model, max_length=48): # ideally text-davinci-001
        assert self.description is None
        query = 'Excerpt:\n\n... ' + passage.strip() + ' ...\n\nWrite a one-sentence summary of ' + self.name + ' in the context of this story.\n\n' + self.name + ' is'
        for _ in range(5):
            descriptions = gpt3_model([query], num_completions=5, generation_max_length=max_length, modify_prompt=False, model_string=ENTITY_MODEL_STRING)
            descriptions = [d for d in descriptions if len(d.strip()) > 0 and len(gpt3_model.tokenizer.encode(d)) < max_length]
            if len(descriptions) > 0:
                break
        if len(descriptions) == 0: # give up
            logging.log(23, 'Warning: Failed to generate sufficiently short description for ' + self.name)
            descriptions = gpt3_model([query], num_completions=1, generation_max_length=max_length, modify_prompt=False, model_string=ENTITY_MODEL_STRING)
        self.description = self.name + ' is' + descriptions[0]
        return self.description
    
    @torch.no_grad()
    def infer_is_character(self, passage, gpt3_model, threshold=CHARACTER_THRESHOLD): # ideally text-davinci-002
        assert self.is_character is None
        query = 'Excerpt:\n\n... ' + passage.strip() + ' ...\n\n' + 'Question:\n\nIs __CHARACTER__ a character, as opposed to e.g., a place or thing?'.replace('__CHARACTER__', self.name) + '\n\n'
        retry = True
        logging.log(21, 'GPT3 CALL' + ' ' + gpt3_model.model + ' ' + str(len(gpt3_model.tokenizer.encode(query)) + 1))
        while retry:
            try:
                completion = openai.Completion.create(
                                    engine=gpt3_model.model,
                                    prompt=query,
                                    max_tokens=1,
                                    temperature=1,
                                    top_p=1,
                                    frequency_penalty=0.5,
                                    presence_penalty=0,
                                    logit_bias={5297: 50, 2949: 50}, # 'Yes' and 'No' for GPT3
                                    logprobs=2)
                retry = False
            except Exception as e: 
                logging.log(23, str(e))
                time.sleep(0.2)
                logging.log(23, 'retrying...')
        logprobs = completion['choices'][0]['logprobs']['top_logprobs'][0]
        logprobs = [logprobs['No'], logprobs['Yes']]
        self.is_character = softmax(logprobs)[1] > threshold
        return self.is_character

    @torch.no_grad()
    def infer_attributes(self, passage, gpt3_model, num_samples=3, detect_contradictions=True, other_names=[], agreement_threshold=2, return_contradiction_prob=False):
        if self.is_character is None or not self.is_character:
            return {}
        logging.log(23, 'INFERRING FOR ' + self.name)
        prompt = passage.strip() + '\n\nQuestion: List very brief facts about __CHARACTER__\'s appearance, personality, and relationship to other characters.\n\n1. __CHARACTER__'.replace('__CHARACTER__', self.name)
        facts_strings = gpt3_model([prompt], num_completions=num_samples, modify_prompt=False, logit_bias={679:-100, 1375:-100, 3347:-100, 1544:-100}, top_p=1, model_string=STRONGER_ENTITY_MODEL_STRING) # ban " He", "He", " She", "She" to force model to refer by name at the beginning of a new entry, to avoid coreference issues
        facts_strings = ['1. ' + self.name + s for s in facts_strings]
        logging.log(22, 'facts strings:' + ' ' + str(facts_strings))
        facts = sum([split_list(s, strict=True) for s in facts_strings], [])
        facts = [s for s in facts if len(s.strip()) > 0]
        facts = [split_paragraphs(s, mode='sentence')[0] for s in facts] # cutoff extra sentences when it's too long
        facts = [f for f in facts if any([tok in f for tok in self.name.split() if tok not in ' '.join(other_names)])]
        logging.log(22, 'facts' + ' ' + str(facts))
        facts = get_agreed_facts(facts, agreement_threshold=agreement_threshold-1)
        logging.log(22, 'agreed facts' + ' ' + str(facts))
        fact_entailment_counts = get_entailment_groups({fact: 1 for fact in facts})
        facts = sorted(list(fact_entailment_counts.keys()), key=lambda x: fact_entailment_counts[x], reverse=True) # most "agreed" facts first
        logging.log(22, 'facts entailment groups' + ' ' + str(facts))
        contradictions = {}
        done_keys = set()
        contradiction_prob = 0
        for fact in facts:
            logging.log(23, 'FACT' + ' ' + fact)
            prompt = construct_infobox_prompt(fact, self.name, other_names)
            infobox_samples = gpt3_model([prompt], num_completions=num_samples, stop='----', logit_bias={50256:-5}, top_p=1, model_string=ENTITY_MODEL_STRING)
            infobox_samples = ['\n' + self.name + s for s in infobox_samples]
            time.sleep(0.5) # otherwise we get rate limited...
            infobox_samples = [s for s in infobox_samples if s.startswith('\n')]
            infobox_samples = [s.strip() for s in infobox_samples if len(s.strip()) > 0]
            if len(infobox_samples) == 0:
                continue
            infobox_keys = Counter()
            for info_sample in infobox_samples:
                for info_entry in info_sample.strip().split('\n'):
                    if info_entry.startswith(self.name):
                        key = split_key_entry(info_entry, self.name)[0]
                        if key is not None:
                            resolved_key = resolve_names(key, [self.name] + other_names)
                            if resolved_key not in self.attributes or self.attributes[resolved_key]['entailment'] < self.ENTAILMENT_RECHECK_THRESHOLD or resolved_key not in done_keys: # don't recompute attributes we already did for this passage
                                infobox_keys[key] += 1
            keys = list(infobox_keys.keys())
            for k in keys:
                if any([banned_attribute in k for banned_attribute in self.BANNED_ATTRIBUTES]):
                    del infobox_keys[k]
            infobox_counts = Counter({self.resample_entry(key, fact, gpt3_model, split_entry=False): infobox_keys[key] for key in infobox_keys}) # resample the values for those keys
            if None in infobox_counts: # resample_entry returns None when it fails
                del infobox_counts[None]
                # logging.log(23, 'DEDUP COUNTS', deduplicated_infobox_entries)
            for infobox_entry in infobox_counts:
                logging.log(23, 'CHECKING' + ' ' + infobox_entry)
                if not infobox_entry.startswith(self.name):
                    continue
                if any([tok in infobox_entry and tok not in fact for tok in sum([other_name.strip().split() for other_name in other_names], [])]):
                    continue # somehow hallucinated a name...??
                new_key, new_entry = split_key_entry(infobox_entry, self.name)
                if new_key is None:
                    logging.log(23, 'Warning: malformed infobox entry' + ' ' + infobox_entry)
                    continue
                if any([bad_word in new_entry.lower() or bad_word in new_key.lower() for bad_word in self.NO_ANSWER_WORDS]):
                    continue
                logging.log(23, infobox_entry + ' ' + new_key + ' ' + new_entry)

                # effectively, ensemble against the QA model to remove some hallucinations
                new_qa_entry_passage, new_qa_score_passage = qa_entailment_best_answer(self.create_entailment_input(new_key, self.name, ''), passage, num_beams=5)
                new_qa_entry_fact, new_qa_score_fact = qa_entailment_best_answer(self.create_entailment_input(new_key, self.name, ''), fact, num_beams=5)

                logging.log(23, 'new_qa_entry_passage' + ' ' + new_qa_entry_passage + ' ' + str(new_qa_score_passage))
                logging.log(23, 'new_qa_entry_fact' + ' ' + new_qa_entry_fact + ' ' + str(new_qa_score_fact))
                logging.log(23, 'min QA confidence' + ' ' + str(min(new_qa_score_passage, new_qa_score_fact)))
                if new_qa_score_passage < self.QA_SCORE_THRESHOLD or new_qa_score_fact < self.QA_SCORE_THRESHOLD or any([w in new_qa_entry_fact.lower() for w in self.NO_ANSWER_WORDS]):
                    logging.log(23, 'filtered by QA confidence')
                    continue
                # make sure we didn't just hallucinate something out of the fact
                fact_entailment_scores, _ = score_entailment([fact], [self.create_entailment_input(new_key, self.get_referred_name(fact), new_entry)])
                fact_entailment_probs = softmax(fact_entailment_scores, axis=1)
                logging.log(23, 'fact entailment' + ' ' + str(fact_entailment_probs[0, 2]))
                if fact_entailment_probs[0, 2] < self.FACT_ENTAILMENT_THRESHOLD:
                    logging.log(23, 'filtered by fact entailment')
                    continue
                
                new_key = resolve_names(new_key, [self.name] + other_names)
                new_entry = resolve_names(new_entry, [self.name] + other_names)
                logging.log(23, 'PASSED FILTERS' + ' ' + self.name + ' ' + new_key + ' ' + new_entry)
                info_dict = {'text': fact, 'entry': new_entry, 'entailment': fact_entailment_probs[0, 2], 'newly_entered': True}
                if return_contradiction_prob:
                    new_prob, _ = self.add_if_better(new_key, info_dict, detect_contradictions=detect_contradictions, contradictions=contradictions, return_contradiction_prob=return_contradiction_prob)
                    contradiction_prob = max(contradiction_prob, new_prob)
                else:
                    contradictions, _ = self.add_if_better(new_key, info_dict, detect_contradictions=detect_contradictions, contradictions=contradictions, return_contradiction_prob=return_contradiction_prob)
                done_keys.add(new_key) # regardless of what happened, no need to check against this key again for the same passage

        if return_contradiction_prob:
            return contradiction_prob
        else:
            return contradictions

    def add_if_better(self, new_key, info_dict, detect_contradictions=True, contradictions=None, return_contradiction_prob=False):
        fact = info_dict['text']
        new_entry = info_dict['entry']
        entailment_prob = info_dict['entailment']
        new_entailment_input = self.create_entailment_input(new_key, self.name, info_dict['entry'])
        status = 'none'
        contradiction_prob = 0
        if new_key in self.attributes:
            original_entailment_input = self.create_entailment_input(new_key, self.name, self.attributes[new_key]['entry'])
            if entailment_equals(new_entailment_input, original_entailment_input): # new one is more detailed
                self.attributes[new_key]['text'] = fact
                self.attributes[new_key]['entry'] = new_entry
                self.attributes[new_key]['entailment'] = entailment_prob
                status = 'modified'
            elif entailment_equals(original_entailment_input, new_entailment_input): # original one is more detailed
                pass
            elif consistent_equals(original_entailment_input, new_entailment_input): # they're consistent with each other, at least
                if fact not in self.attributes[new_key]['text']:
                    self.attributes[new_key]['text'] += '\n' + fact
                if new_entry not in self.attributes[new_key]['entry']:
                    self.attributes[new_key]['entry'] += ', ' + new_entry
                    status = 'modified'
                if not self.attributes[new_key]['newly_entered']:
                    contradiction_prob = math.exp(score_entailment(original_entailment_input, new_entailment_input)[0][0, 0])
            elif self.attributes[new_key]['newly_entered']: # both part of the same passage, so just pick what's higher confidence
                if entailment_prob > self.attributes[new_key]['entailment']:
                    logging.log(23, 'CHANGED BY RECHECK')
                    logging.log(23, 'old' + ' ' + self.attributes[new_key]['text'] + ' ' + self.attributes[new_key]['entry'] + ' ' + str(self.attributes[new_key]['entailment']))
                    self.attributes[new_key]['text'] = fact
                    self.attributes[new_key]['entry'] = new_entry
                    self.attributes[new_key]['entailment'] = entailment_prob
                    logging.log(23, 'new' + ' ' + self.attributes[new_key]['text'] + ' ' + self.attributes[new_key]['entry'] + ' ' + str(self.attributes[new_key]['entailment']))
                    status = 'modified'
            elif not detect_contradictions:
                # if not detect_contradictions: presumably fixed elsewhere.
                pass
            else:
                logging.log(23, 'POTENTIAL CONTRADICTION')
                logging.log(23, 'ENTITY' + ' ' + self.name)
                logging.log(23, 'KEY' + ' ' + new_key)
                logging.log(23, 'EXISTING ENTRY' + ' ' + str(self.attributes[new_key]))
                logging.log(23, 'NEW ENTRY' + ' ' + str({'text': fact, 'entry': new_entry}))
                contradictions[new_key] = (self.attributes[new_key], {'text': fact, 'entry': new_entry})
                contradiction_prob = math.exp(score_entailment(original_entailment_input, new_entailment_input)[0][0, 0])
            if return_contradiction_prob:
                return contradiction_prob, status
        else:
            self.attributes[new_key] = info_dict
            status = 'added'
            if return_contradiction_prob:
                return contradiction_prob, status
        return contradictions, status