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

class BeamCandidate:
    def __init__(self, 
                 args, 
                 all_entities_dict,
                 infer_attributes_string,
                 model=None,
                 controllers=None,
                 step=0, 
                 alignment_score=-1e8, 
                 best_alignment_so_far=-1e8,
                 all_paragraphs=[], 
                 outline_sections=[],
                 paragraphs_by_outline_section=None):
        self.args = args
        self.all_entities_dict = all_entities_dict
        self.infer_attributes_string = infer_attributes_string
        self.model = model
        self.controllers = controllers
        self.step = step
        self.alignment_score = alignment_score
        self.best_alignment_so_far = best_alignment_so_far
        self.all_paragraphs = all_paragraphs
        self.outline_sections = outline_sections
        self.paragraphs_by_outline_section = paragraphs_by_outline_section if paragraphs_by_outline_section is not None else {}
        self.is_consistent = False
    
    def story(self):
        # technically there could be a missing newline here instead of a space. low priority
        return ' '.join(self.all_paragraphs)
    
    def previous_passage(self, max_tokens, suffix=None):
        if len(self.all_paragraphs) == 0:
            return ''
        passage = self.story()
        if len(self.story().strip()) == 0:
            return ''
        if suffix is not None:
            passage = passage[:len(passage) - len(suffix)].rstrip()
        if len(passage.strip()) == 0:
            return ''
        passage = self.model.tokenizer.decode(self.model.tokenizer.encode(passage)[-max_tokens:])
        return cut_first_sentence(passage)
    
    def print_section(self, section_idx):
        return ' '.join(self.paragraphs_by_outline_section[self.outline_sections[section_idx]])

    def select_entities(self, outline_section, previous_paragraph=None):
        # TODO lot of things to try here...
        matched_entities, _, _ = deduplicate_match_entities(detect_entities(outline_section), self.all_entities_dict.keys())
        matched_entities = list(matched_entities)
        dpr_query_encoder, dpr_context_encoder = load_dpr()
        if previous_paragraph is not None:
            summary_prompt = previous_paragraph.strip() + '\n\n\n\nOne-sentence summary:\n\n\n\n'
            summary = self.model([summary_prompt], modify_prompt=False, model_string='text-curie-001')[0].strip()
            query_encoding = dpr_query_encoder.encode('Previous passage summary: ' + summary + '\n\nCurrent story outline: ' + outline_section.strip() + '\n\nWho or what appears in the upcoming paragraphs?')
        else:
            query_encoding = dpr_query_encoder.encode('Current story outline: ' + outline_section.strip() + '\n\nWho or what appears in the upcoming paragraphs?')
        entities = [key for key in list(self.all_entities_dict.keys()) if self.all_entities_dict[key].is_character]
        if len(entities) == 0:
            return []
        context_encodings = dpr_context_encoder.encode(entities)
        scores = (query_encoding.reshape(1, -1) * context_encodings).sum(axis=1)
        selected_entities = matched_entities
        total_tokens = sum([len(self.model.tokenizer.encode(entity)) for entity in selected_entities])
        if total_tokens > self.args.max_entity_context_tokens:
            logging.warning('Warning: truncating entity context to fit context length limit')
            selected_entities = []
            total_tokens = 0
            for entity in matched_entities:
                total_tokens += len(self.model.tokenizer.encode(entity))
                if total_tokens > self.args.max_entity_context_tokens:
                    break
                selected_entities.append(entity)
            return selected_entities
        # sample additional entities without repeats, up to context length
        for i, ent in enumerate(entities): # mask out ones we already selected
            if ent in selected_entities:
                scores[i] = -1e8
        unselected_entities = [ent for ent in entities if ent not in selected_entities]
        while total_tokens < self.args.max_entity_context_tokens and len(unselected_entities) > 0:
            probs = softmax(scores)
            next_entity = np.random.choice(entities, p=probs)
            total_tokens += len(self.model.tokenizer.encode(next_entity))
            if total_tokens > self.args.max_entity_context_tokens:
                break
            selected_entities.append(next_entity)
            scores[entities.index(next_entity)] = -1e8
            unselected_entities = [ent for ent in entities if ent not in selected_entities]

        return selected_entities
    
    def create_updated_entities(self, new_passage, cached_update_dict=None):
        # detect and make entries for new entities, run inference for description / is_character on new entities, update attributes
        new_entities_dict = deepcopy(self.all_entities_dict)
        entities = [str(ent) for ent in detect_entities(new_passage)]
        matched_entities, new_entities, _ = deduplicate_match_entities(entities, self.all_entities_dict.keys())
        new_entities_dict = deepcopy(self.all_entities_dict)
        for ent in new_entities:
            entity = Entity(ent)
            entity.infer_description(new_passage, self.model, max_length=self.args.entity_description_max_length)
            entity.infer_is_character(new_passage, self.model)
            entity.infer_attributes(new_passage, self.model, other_names=[name for name in matched_entities if name != entity.name] + [name for name in new_entities if name != entity.name])
            new_entities_dict[ent] = entity
        for ent in matched_entities:
            if cached_update_dict is not None and ent in cached_update_dict:
                new_entities_dict[ent] = cached_update_dict[ent]
            else:
                new_entities_dict[ent].infer_attributes(new_passage, self.model, other_names=[name for name in matched_entities if name != ent] + list(new_entities), detect_contradictions=False)
        complete_mutual_relations(new_entities_dict, self.model)
        return new_entities_dict
    
    def detect_attribute_contradictions(self, completion, detect_contradictions=True):
        matched_entities, new_entities, _ = deduplicate_match_entities(detect_entities(completion, add_dpr_entities=False, all_entities_dict=self.all_entities_dict), self.all_entities_dict.keys())
        matched_entities = list(matched_entities)
        contradictions = {}
        cached_update_dict = {}
        copied_entities = deepcopy(self.all_entities_dict)
        for ent in matched_entities:
            entity = copied_entities[ent]
            contradictions[ent] = entity.infer_attributes(completion, self.model, detect_contradictions=detect_contradictions, other_names=[name for name in matched_entities if name != entity.name] + list(new_entities))
            cached_update_dict[ent] = entity
        _, additional_contradictions = complete_mutual_relations(copied_entities, self.model)
        for ent in additional_contradictions:
            for key in additional_contradictions[ent]:
                if ent not in contradictions:
                    contradictions[ent] = {}
                contradictions[ent][key] = additional_contradictions[ent][key]
        return matched_entities, contradictions, cached_update_dict

    def condense_outline_sections(self, outline):
        if type(outline) != tuple:
            return
        logging.log(23, 'CONDENSING OUTLINE')
        logging.log(23, 'BEFORE')
        logging.log(23, str(self.outline_sections))
        high_level_outline = split_list(outline[0])
        for i in range(len(high_level_outline)):
            if high_level_outline[i] in self.outline_sections:
                assert self.outline_sections[i] == high_level_outline[i]
                continue
            detailed_outline = split_list(outline[1][i])
            if len(self.outline_sections) - i == len(detailed_outline):
                self.outline_sections = deepcopy(high_level_outline[:i+1])
            break
        logging.log(23, 'AFTER')
        logging.log(23, str(self.outline_sections))

    def construct_prompt(self, outline_section, selected_entities=[]):
        presumed_max_prompt_length = 2*self.args.generation_max_length + self.args.max_entity_context_tokens + 128
        if self.args.no_planner:
            if len(self.model.tokenizer.encode(self.story())) <= self.args.max_context_length - 2*self.args.generation_max_length: # early on enough to fit the premise in the rolling window
                prompt = 'Write a story with the following premise.\n\n' + self.all_entities_dict['Premise'].description + '\n\n'
                prompt += 'Chapter 1\n\n'
                if len(self.story()) > 0:
                    prompt += self.story()
                return prompt
            else:
                return self.previous_passage(self.args.max_context_length - self.args.generation_max_length)
        if len(self.all_paragraphs) == 0:
            prompt = self.infer_attributes_string + '\n\n\n\n'
        else:
            if len(selected_entities) > 0:
                selected_entity_strings = [self.all_entities_dict[ent].description for ent in selected_entities]
                prompt = 'Relevant Context:\n\n' + '\n\n'.join(selected_entity_strings) + '\n\n\n\n'
        prompt += 'The story is written in third person.'
        if self.step > 1:
            prompt += '\n\n\n\nPrevious story summary: ' + ' '.join(self.outline_sections[:-1])
        previous_text = self.previous_passage(self.args.previous_prompt_length)
        if len(self.all_paragraphs) > 0:
            previous_passage = self.previous_passage(int(self.args.max_context_length/2), suffix=previous_text)
            if len(self.model.tokenizer.encode(previous_passage)) > int(self.args.max_context_length/4): # no need to do this extra summary if it's really short
                max_preceding_summary_tokens = 128
                preceding_summary = self.model([previous_passage + '\n\nSummarize the events in this passage.'], generation_max_length=max_preceding_summary_tokens, model_string='text-curie-001')[0].strip()
                if len(self.model.tokenizer.encode(preceding_summary)) == max_preceding_summary_tokens:
                    logging.warning('Warning: preceding events summary is too long, truncating')
                prompt += '\n\n\n\nEvents immediately prior to the upcoming passage: ' + preceding_summary
        if self.step == 1:
            prompt += '\n\n\n\nChapter 1 Summary: ' + outline_section.strip()
        else:
            prompt += '\n\n\n\nIn the upcoming passage, ' + outline_section.strip()[0].lower() + outline_section.strip()[1:] # uncapitalize the first letter if needed
        prompt += '\n\n\n\nFull text below:\n\n\n\n'
        if len(self.all_paragraphs) == 0:
            prompt = prompt + 'Chapter 1\n\n'
        prompt = prompt + previous_text
        if len(self.model.tokenizer.encode(prompt)) > presumed_max_prompt_length:
            # generation max length from selected entities and outline, max entity context tokens from previous context, then some padding
            logging.warning('Warning: prompt is too long, please inspect')
            import pdb; pdb.set_trace()
        return prompt
    
    @torch.no_grad()
    def edit_update_contradictions(self):
        assert not self.is_consistent
        completion = self.all_paragraphs[-1]
        autoregressive_context = self.all_paragraphs[-2].lstrip(string.punctuation) if len(self.all_paragraphs) > 1 else ''
        matched_entities, contradictions, cached_update_dict = self.detect_attribute_contradictions(completion.strip(), detect_contradictions=True)
        edited_sentences = set()
        if any([len(contradictions[ent]) > 0 for ent in matched_entities]) and len(autoregressive_context) > 0: # don't do it on the first paragraph, if we don't have autoregressive context to help check we're not messing something up
            logging.log(23, 'editing completion based on contradictions')
            logging.log(23, 'AUTOREGRESSIVE CONTEXT ' + autoregressive_context)
            logging.log(23, 'BEFORE ' + completion)
            for ent in matched_entities:
                for contradiction_key in contradictions[ent]:
                    for contradicted_sentence in contradictions[ent][contradiction_key][0]['text'].strip().split('\n'):
                        if contradicted_sentence in edited_sentences: # no need to edit again if the sentence was contradicted more than once
                            continue
                        edited_sentences.add(contradicted_sentence)
                        instruction = 'Edit so that ' + contradicted_sentence + ' Keep the text unchanged as much as possible.'
                        logging.log(23, 'INSTRUCTION ' + instruction)
                        completion = gpt3_edit(completion, instruction, prefix=None if len(autoregressive_context.strip()) == 0 else autoregressive_context).strip()
                        if len(self.model.tokenizer.encode(completion)) > self.args.generation_max_length + 64: # give some leeway for editing to expand text
                            logging.warning('WARNING: completion is too long after editing. Truncating...')
                            completion = self.model.tokenizer.decode(self.model.tokenizer.encode(completion)[:self.args.generation_max_length + 64])
                            completion = cut_last_sentence(completion)
            logging.log(23, 'AFTER ' + completion)
            _, _, cached_update_dict = self.detect_attribute_contradictions(completion.strip(), detect_contradictions=False) # only reupdate the cache, and allow appending any new entries; presumably GPT3 fixed any "real" contradictions
        self.all_paragraphs[-1] = completion
        self.paragraphs_by_outline_section[self.outline_sections[-1]][-1] = completion
        self.all_entities_dict = self.create_updated_entities(completion.strip(), cached_update_dict=cached_update_dict)
        self.is_consistent = True

    @torch.no_grad()
    def extend(self, outline_section):
        # return a list of up to max_beam_size new BeamCandidates with their respective alignment scores before moving on to the next outline sentence
        logging.log(25, 'extension step ' + str(self.step))
        self.step += 1
        self.alignment_score = -1e8
        self.best_alignment_so_far = -1e8
        self.outline_sections.append(outline_section)
        self.paragraphs_by_outline_section[outline_section] = []
        completed_candidates = []
        beam = [self]
        substep = 0
        while len(completed_candidates) < self.args.max_beam_size:
            logging.log(25, 'substep ' + str(substep))
            next_candidates = []
            for beam_idx, prev_candidate in enumerate(beam):
                candidates = []
                for candidate in prev_candidate.extend_single(outline_section, batch_size=self.args.max_candidates, top_p=self.args.draft_top_p):
                    candidates.append(candidate)
                    logging.log(25, 'beam idx ' + str(beam_idx) + ' single extension with score ' + str(candidates[-1].alignment_score))
                candidates = sorted(candidates, key=lambda x: x.alignment_score, reverse=True)
                if candidates[0].alignment_score < prev_candidate.best_alignment_so_far - self.args.continuation_threshold: # early termination of expansion of this outline point
                    logging.log(25, 'beam idx ' + str(beam_idx) + ' adding completed candidate with score ' + str(prev_candidate.alignment_score))
                    assert self.args.no_editor or prev_candidate.is_consistent
                    completed_candidates.append(prev_candidate)
                else:
                    if candidates[0].alignment_score < prev_candidate.best_alignment_so_far:
                        logging.log(25, 'continuation with slightly worse score')
                    next_candidates.extend(candidates)
            next_candidates = sorted(next_candidates, key=lambda x: x.alignment_score, reverse=True)[:self.args.max_beam_size - len(completed_candidates)]
            beam = next_candidates
            if not self.args.no_editor:
                for c in beam:
                    c.edit_update_contradictions()
            substep += 1
            if substep >= self.args.max_continuation_substeps: # fill out the rest of the completed candidates
                for c in beam:
                    logging.log(25, 'beam idx ' + str(beam_idx) + ' adding completed candidate with score ' + str(c.alignment_score))
                    assert self.args.no_editor or c.is_consistent
                    completed_candidates.append(c)
                break
        return sorted(completed_candidates, key=lambda x: x.alignment_score, reverse=True)[:self.args.max_beam_size]
    
    def calculate_alignment(self, completions, prompt, outline_section):
        if self.args.max_candidates == 1:
            return np.zeros(len(completions)) # in this case, we're doing no reranking, and this will also prevent the reranking from being used to decide when to stop. 
        repetition_penalty = np.array([calculate_repetition_length_penalty(c, [prompt]) for c in completions])
        is_first_person = np.array([1 if detect_first_second_person(c) else 0 for c in completions]) # could have some false positives if the quotations are off, but whatever.
        repetition_penalty += is_first_person * 10
        alignment_score = 0
        if not self.args.no_planner:
            alignment_input = [create_prefix_completion(c, outline_section)[1] for c in completions]
            alignment_score = self.controllers[0].evaluate_overall_texts(alignment_input).cpu().numpy() # logprob for alignment with outline
        if len(self.story().strip()) > 0:
            alignment_score += self.controllers[1]([cut_first_sentence(self.previous_passage(1000)) for _ in range(len(completions))], completions).cpu().numpy() # logprob for alignment with previous story, up to 1k prev tokens
        alignment_score += -repetition_penalty * self.args.repetition_penalty_weight
        return alignment_score

    def extend_single(self, outline_section, batch_size=1, top_p=None):
        if self.args.outline_levels == 1:
            assert self.step == len(self.outline_sections)
        if self.args.no_planner:
            selected_entities = None
        else:
            selected_entities = self.select_entities(outline_section, previous_paragraph=self.all_paragraphs[-1] if len(self.all_paragraphs) > 0 else None)
        prompt = self.construct_prompt(outline_section, selected_entities=selected_entities)
        logging.log(21, 'PROMPT')
        logging.log(21, prompt)
        completions = self.model([prompt], model_string=self.args.draft_model_string, modify_prompt=False, num_completions=batch_size, top_p=top_p, temperature=self.args.summarizer_temperature, cut_sentence=True, logit_bias={50256:-100, 14126:-100, 7006:-100, 6843:-100, 43582:-100}) # don't let it end prematurely, and don't let it repeatedly generate variants of the word "chapter" since we used it to prompt it initially # stop=['Chapter', 'Chapters', 'Full text', '\n\n\n\n\n']
        for i in range(len(completions)):
            completions[i] = completions[i].strip()
            while '\n\n\n' in completions[i]: # just improve the formatting a bit
                completions[i] = completions[i].replace('\n\n\n', '\n\n')
        for i in range(len(completions)):
            _, _, replacements = deduplicate_match_entities(detect_entities(completions[i]), self.all_entities_dict.keys())
            if not self.args.no_editor:
                for key, value in replacements.items():
                    completions[i] = completions[i].replace(key, value)
        alignment_score = self.calculate_alignment(completions, prompt, outline_section)
        new_candidates = []
        for c, s in zip(completions, alignment_score):
            new_paragraphs_by_outline_section = deepcopy(self.paragraphs_by_outline_section)
            new_paragraphs_by_outline_section[outline_section].append(c)
            new_candidates.append(BeamCandidate(self.args, 
                                self.all_entities_dict,
                                self.infer_attributes_string,
                                model=self.model, 
                                controllers=self.controllers, 
                                step=self.step, 
                                alignment_score=s, 
                                best_alignment_so_far=max(s, self.best_alignment_so_far),
                                all_paragraphs=deepcopy(self.all_paragraphs) + [c], 
                                outline_sections=deepcopy(self.outline_sections),
                                paragraphs_by_outline_section=new_paragraphs_by_outline_section))
        return new_candidates
    
    def complete_ending(self):
        outline_section = self.outline_sections[-1]
        if self.args.no_planner:
            selected_entities = None
        else:
            selected_entities = self.select_entities(outline_section, previous_paragraph=self.all_paragraphs[-1] if len(self.all_paragraphs) > 0 else None)
        prompt = self.construct_prompt(outline_section, selected_entities=selected_entities)
        completions = gpt3_insert(prompt, 
                                 '\n\n\n\n' + GPT3_END, 
                                 top_p=self.args.draft_top_p, 
                                 temperature=self.args.summarizer_temperature,
                                 n=self.args.max_candidates, 
                                 max_tokens=self.args.generation_max_length, 
                                 frequency_penalty=self.args.summarizer_frequency_penalty,
                                 presence_penalty=self.args.summarizer_presence_penalty)
        completions = [c.replace('\n\n\n\n', '\n\n') for c in completions]
        alignment_score = self.calculate_alignment(completions, prompt, outline_section)
        logging.log(23, 'ENDING ALIGNMENT SCORES ' + str(alignment_score))
        ranked_completions = sorted(zip(completions, alignment_score), key=lambda x: x[1], reverse=True)
        ending = ranked_completions[0][0]
        should_continue = len(self.model.tokenizer.encode(ending))==self.args.generation_max_length # ending didn't finish writing; should generate more toward the ending after this
        ending = cut_last_sentence(ending)
        logging.log(23, 'ENDING' + ' ' + ending)
        new_paragraphs_by_outline_section = deepcopy(self.paragraphs_by_outline_section)
        if outline_section not in new_paragraphs_by_outline_section:
            new_paragraphs_by_outline_section[outline_section] = []
        new_paragraphs_by_outline_section[outline_section].append(ending)
        new_candidate = BeamCandidate(self.args, 
                            self.all_entities_dict,
                            self.infer_attributes_string,
                            model=self.model, 
                            controllers=self.controllers, 
                            step=self.step, 
                            alignment_score=self.alignment_score, 
                            best_alignment_so_far=self.best_alignment_so_far,
                            all_paragraphs=deepcopy(self.all_paragraphs) + [ending], 
                            outline_sections=deepcopy(self.outline_sections),
                            paragraphs_by_outline_section=new_paragraphs_by_outline_section)
        if not self.args.no_editor:
            new_candidate.edit_update_contradictions()
        return new_candidate, should_continue