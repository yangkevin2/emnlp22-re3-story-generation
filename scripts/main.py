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
import json

import torch
import Levenshtein
import numpy as np
from transformers import AutoTokenizer
import openai
from scipy.special import softmax

from story_generation.edit_module.entity import *
from story_generation.draft_module.beam_candidate import BeamCandidate
from story_generation.plan_module.plan import *
from story_generation.common.util import *
from story_generation.common.data.data_util import add_data_args, load_dataset
from story_generation.common.summarizer.summarizer_util import add_summarizer_args, load_summarizer
from story_generation.common.summarizer.models.gpt3_summarizer import GPT3_SEP, GPT3_END
from story_generation.common.summarizer.models.opt_summarizer import OPTSummarizer
from story_generation.common.controller.controller_util import add_controller_args, load_controller
from story_generation.common.controller.loaders.alignment_loader import create_prefix_completion
from story_generation.common.data.split_paragraphs import *


if __name__=='__main__':
    parser = argparse.ArgumentParser() # parameter defaults are set to values used in paper
    parser = add_general_args(parser)
    parser = add_data_args(parser)
    parser = add_summarizer_args(parser)
    parser = add_controller_args(parser)

    parser.add_argument('--premise', type=str, default=None, help='Premise to use for generation')

    # SAVE/LOAD PLAN/LOGS
    parser.add_argument('--load-outline-file', type=str, help='load outline from this file')
    parser.add_argument('--save-outline-file', type=str, help='save outline to this file')
    parser.add_argument('--save-complete-file', type=str, help='save completed beam object to this file')
    parser.add_argument('--log-file', type=str, help='logging file', default=None)
    parser.add_argument('--log-level', type=int, default=22, help='logging level; decrease to 21 for full verbosity while suppressing stuff openai and urllib')

    # ALTERNATE MODES / ABLATIONS
    parser.add_argument('--setup-only', action='store_true', help='exit after generating the premise/setup/outline')
    parser.add_argument('--no-attributes', action='store_true', help='do not infer attributes')
    parser.add_argument('--no-editor', action='store_true', help='do not use editor to edit text for detected contradictions')
    parser.add_argument('--no-planner', action='store_true', help='do not planner beyond the initial setup')

    # SEARCH SIZE / BEAM PARAMETERS
    parser.add_argument('--max-candidates', type=int, default=10, help='max number of candidates to generate at each step by each beam candidate')
    parser.add_argument('--max-beam-size', type=int, default=1, help='max number of beam candidates to generate at each step')
    parser.add_argument('--beam-max-difference', type=float, default=1, help='max difference between beam scores')

    # OUTLINE PARAMETERS
    parser.add_argument('--fixed-outline-length', type=int, default=3, help='fixed length for outline; use -1 for no fixed length')
    parser.add_argument('--outline-levels', type=int, default=1, help='num levels of hierarchy in outline')

    # CONTINUATION ALIGNMENT / LENGTH PARAMETERS
    parser.add_argument('--continuation-threshold', type=float, default=10000, help='if alignment score is worse by at least this much, move on to next outline point; 10000 basically turns this off')
    parser.add_argument('--max-continuation-substeps', type=int, default=4, help='max number of continuation candidates to generate at each step')
    parser.add_argument('--max-ending-continuations', type=int, default=3, help='max number of continuation steps for ending the story')

    # PROMPT PARAMETERS
    parser.add_argument('--previous-prompt-length', type=int, default=256, help='length of previously generated text in prompt')
    parser.add_argument('--max-entity-context-tokens', type=int, default=128, help='max number of tokens to use for entity context')
    parser.add_argument('--entity-description-max-length', type=int, default=48, help='max number of tokens to use per entity description')
    
    # GENERATION PARAMETERS
    parser.add_argument('--extension-method', type=str, choices=['gpt3', 'opt'], default='gpt3', help='generator to use for main story drafting')
    parser.add_argument('--repetition-penalty-weight', type=float, default=5, help='weight of repetition penalty')
    parser.add_argument('--draft-top-p', type=float, default=1, help='initial top_p for beam search')
    parser.add_argument('--plan-model-string', type=str, default='text-davinci-002', help='gpt3 model string to use in planning')
    parser.add_argument('--draft-model-string', type=str, default='davinci', help='gpt3 model string to use in extending story')
    parser.add_argument('--cut-sentence', action='store_true', default=False, help='cut incomplete sentence at end of generation')
    
    args = parser.parse_args()

    if os.path.exists(args.save_complete_file):
        logging.log(25, 'save file already exists')
        sys.exit()

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    logging.basicConfig(format='%(message)s', filename=args.log_file, level=args.log_level)

    gpt3_model = load_summarizer(args) # naming is a relic of some old preliminary experiments; it's just a gpt3 interface
    controllers = [load_controller(args, i) for i in range(len(args.controller))]
    assert all([controller.type == 'sentence' for controller in controllers])
    opt_model = OPTSummarizer(args) if args.extension_method == 'opt' else None
    
    if args.load_outline_file is not None:
        if args.load_outline_file.endswith('.pkl'):
            save_info = load_plan_info(args.load_outline_file)
        else:
            with open(args.load_outline_file, 'r') as f:
                info = json.load(f)
                if 'character_strings' not in info:
                    character_strings = {}
                    for name, desc in info['character_info']:
                        character_strings[name] = Entity(name, desc, is_character=True)
                save_info = {'premise': info['premise'],
                            'setting': info['setting'],
                            'character_strings': character_strings,
                            'outline': None,
                            'outline_sections': info['outline_sections'],
                            'infer_attributes_string': info['premise'] + '\n\n' + info['setting'] + '\n\n' + '\n\n'.join([c.description for c in character_strings.values()])}
        if (not args.no_attributes and not args.no_editor and not args.no_planner): # fill in the attributes if we need them, if they're not already present in the save
            infer_initial_attributes_from_plan(save_info, gpt3_model)
    else:
        save_info = generate_plan_info(args, gpt3_model, model_string=args.plan_model_string)
        if args.save_outline_file is not None:
            os.makedirs(os.path.dirname(args.save_outline_file), exist_ok=True)
            with open(args.save_outline_file, 'wb') as f:
                pickle.dump(save_info, f)
    if args.setup_only:
        sys.exit()
    
    premise = save_info['premise']
    setting = save_info['setting']
    character_strings = save_info['character_strings']
    outline_sections = save_info['outline_sections']
    infer_attributes_string = premise + '\n\n' + setting + '\n\n' + '\n\n'.join([c.description for c in character_strings.values()])
    
    if args.no_attributes:
        all_entities_dict = {}
    else:
        all_entities_dict = deepcopy(character_strings)
        all_entities_dict['Premise'] = Entity('Premise', description='Premise: ' + premise.strip(), is_character=False)
        all_entities_dict['Setting'] = Entity('Setting', description='Setting: ' + setting.strip(), is_character=False)

    all_paragraphs = []
    previous_alignment_score = -1e8
    beam = [BeamCandidate(args, 
                          all_entities_dict, 
                          infer_attributes_string,
                          model=gpt3_model,
                          opt_model=opt_model,
                          controllers=controllers)]
    if not args.no_editor and not args.no_planner:
        for candidate in beam:
            candidate.all_entities_dict = candidate.create_updated_entities('\n\n'.join(outline_sections))
    if args.no_planner: # only get the premise
        for candidate in beam:
            initial_keys = list(candidate.all_entities_dict.keys())
            for key in initial_keys:
                if key != 'Premise':
                    del candidate.all_entities_dict[key]
    outline_sections[-1] = outline_sections[-1] + ' This is the end of the story.'

    # restart from intermediate if exists
    for i in range(len(outline_sections)-1, -1, -1):
        if os.path.exists(args.save_complete_file + '.temp' + str(i)):
            logging.log(25, 'found temp file for section ' + str(i) + ', restarting from there')
            with open(args.save_complete_file + '.temp' + str(i), 'rb') as f:
                beam = pickle.load(f)
                for b in beam:
                    b.controllers = controllers
                    b.model = gpt3_model
                    b.opt_model = opt_model
            break

    for i in range(len(outline_sections)):
        logging.log(25, '\n\n\n\niteration at step ' + str(i))
        outline_section = outline_sections[i]
        if outline_section in beam[0].outline_sections:
            logging.log(25, 'already generated this section')
            continue
        extensions = sum([b.extend(outline_section) for b in beam], [])
        extensions = sorted(extensions, key=lambda x: x.best_alignment_so_far, reverse=True)
        # pick the best extension plus up to max_beam_size that are below some alignment threshold
        new_beam = [extensions[0]]
        for extension in extensions[1:args.max_beam_size]:
            if extension.best_alignment_so_far > extensions[0].best_alignment_so_far - args.beam_max_difference: # variable beam size
                new_beam.append(extension)
        beam = new_beam
        for b in beam:
            b.condense_outline_sections(None)
        logging.log(25, '\n\n\n\nend of iteration ' + str(i))
        for entity in beam[0].all_entities_dict.values():
            logging.debug(entity)
        logging.log(23, beam[0].story())

        # save intermediate
        with open(args.save_complete_file + '.temp' + str(i), 'wb') as f:
            for b in beam:
                b.controllers = None
            pickle.dump(beam, f)
            for b in beam:
                b.controllers = controllers
        if i > 0 and os.path.exists(args.save_complete_file + '.temp' + str(i-1)):
            os.remove(args.save_complete_file + '.temp' + str(i-1))
    
    for i in range(len(beam)):
        should_continue = True
        num_attempts = 0
        while should_continue:
            logging.log(25, 'BEAM ' + str(i) + ' ENDING ATTEMPT ' + str(num_attempts))
            beam[i], should_continue = beam[i].complete_ending()
            num_attempts += 1
            if num_attempts >= args.max_ending_continuations:
                break

    logging.log(25, '\n\n\n\nFINAL STORY')
    logging.log(25, beam[0].story())
    if args.save_complete_file is not None:
        with open(args.save_complete_file, 'wb') as wf:
            for b in beam:
                b.controllers = None
            pickle.dump(beam, wf)
