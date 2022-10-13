import argparse
import csv
import os
import time
import pickle
from copy import deepcopy

from transformers import AutoTokenizer

from story_generation.edit_module.entity import Entity
from story_generation.common.util import *
from story_generation.common.data.data_util import add_data_args, load_dataset
from story_generation.common.summarizer.summarizer_util import add_summarizer_args, load_summarizer

def generate_story(model, prompt, character, dpr_query, num_samples=1):
    stories = [character + story for story in model([prompt], stop='Chapter', cut_sentence=True, num_completions=num_samples)]
    scores = score_dpr(dpr_query, stories)
    stories_scores = [(story, score) for story, score in zip(stories, scores)]
    return sorted(stories_scores, key=lambda x: x[1], reverse=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser = add_general_args(parser)
    parser = add_data_args(parser)
    parser = add_summarizer_args(parser)
    parser.add_argument('--load-dir', type=str, required=True, help='directory where stuff was saved in resample_character_descriptions.py')
    args = parser.parse_args()

    base_model = load_summarizer(args)
    instruct_args = deepcopy(args)
    instruct_args.gpt3_model = 'text-' + args.gpt3_model + '-001'
    instruct_model = load_summarizer(instruct_args)

    original_dir = os.path.join(args.load_dir, 'original')
    altered_dir = os.path.join(args.load_dir, 'altered')
    original_stories_dir = os.path.join(args.load_dir, 'original_stories')
    altered_stories_dir = os.path.join(args.load_dir, 'altered_stories')

    os.makedirs(original_stories_dir, exist_ok=True)
    os.makedirs(altered_stories_dir, exist_ok=True)

    num_files = len(os.listdir(original_dir))
    num_preexisting_files = len(os.listdir(original_stories_dir))

    for i in range(num_preexisting_files, num_files): # resume where we left off, if restarting the script
        with open(os.path.join(original_dir, str(i) + '.pkl'), 'rb') as f:
            original_save_info = pickle.load(f)
        with open(os.path.join(altered_dir, str(i) + '.pkl'), 'rb') as f:
            altered_save_info = pickle.load(f)
        modified_character = None
        for character in original_save_info['character_strings']:
            if original_save_info['character_strings'][character].description != altered_save_info['character_strings'][character].description:
                modified_character = character
                break
        assert modified_character is not None
        prompt_modifier = '\n\nWrite a story with the above premise, setting, and characters.\n\nChapter 1\n\n' + modified_character # make it start the story talking about the right character.
        original_prompt = original_save_info['infer_attributes_string'] + prompt_modifier
        altered_prompt = altered_save_info['infer_attributes_string'] + prompt_modifier

        print('ORIGINAL PROMPT')
        print(original_prompt)
        print('ALTERED CHARACTER')
        print(modified_character)
        print('ORIGINAL DESCRIPTION')
        print(original_save_info['character_strings'][modified_character].description)
        print('ALTERED DESCRIPTION')
        print(altered_save_info['character_strings'][modified_character].description)
        print('CONTRADICTED ORIGINAL')
        print(original_save_info['contradicted_part'])
        print('CONTRADICTED ALTERED')
        print(altered_save_info['contradicted_part'])

        while True:
            original_stories = generate_story(base_model, original_prompt, modified_character, original_save_info['contradicted_part'] + '\nFind evidence to support or refute this description.', num_samples=5)
            for original_story, score in original_stories:
                print('\n\nORIGINAL STORY')
                print(original_story)
                print('DPR SCORE')
                print(score)
                is_good = input('Is this story good? (y/n/s) ')
                if is_good == 'y' or is_good == 's':
                    break
            if is_good != 'n':
                break
        if is_good != 'y':
            with open(os.path.join(original_stories_dir, str(i) + '.txt'), 'w') as f:
                f.write('SKIPPED')
            with open(os.path.join(altered_stories_dir, str(i) + '.txt'), 'w') as f:
                f.write('SKIPPED')
            continue
        while True:
            altered_stories = generate_story(base_model, altered_prompt, modified_character, original_save_info['contradicted_part'] + '\nFind evidence to support or refute this description.', num_samples=5)
            for altered_story, score in altered_stories: 
                print('\n\nALTERED STORY')
                print(altered_story)
                print('DPR SCORE')
                print(score)
                is_good = input('Is this story good? (y/n/s) ')
                if is_good == 'y' or is_good == 's':
                    break
            if is_good != 'n':
                break
        if is_good != 'y':
            with open(os.path.join(original_stories_dir, str(i) + '.txt'), 'w') as f:
                f.write('SKIPPED')
            with open(os.path.join(altered_stories_dir, str(i) + '.txt'), 'w') as f:
                f.write('SKIPPED')
            continue
        with open(os.path.join(original_stories_dir, str(i) + '.txt'), 'w') as f:
            f.write(original_story)
        with open(os.path.join(altered_stories_dir, str(i) + '.txt'), 'w') as f:
            f.write(altered_story)

    
        