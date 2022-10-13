import argparse
import pickle
import os
import random
from copy import deepcopy

from story_generation.common.util import *
from story_generation.common.data.data_util import add_data_args, load_dataset
from story_generation.common.summarizer.summarizer_util import add_summarizer_args, load_summarizer
from story_generation.common.controller.controller_util import add_controller_args, load_controller
from story_generation.plan_module.plan import generate_outline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_general_args(parser)
    parser = add_data_args(parser)
    parser = add_summarizer_args(parser)
    parser.add_argument('--load-dir', type=str, required=True, help='directory to load setups from')
    parser.add_argument('--save-dir', type=str, required=True, help='directory to save setups to')
    args = parser.parse_args()

    already_labeled_paths = []
    if os.path.exists(os.path.join(args.save_dir, 'already_labeled.txt')):
        with open(os.path.join(args.save_dir, 'already_labeled.txt'), 'r') as f:
            for line in f:
                already_labeled_paths.append(line.strip())

    base_model = load_summarizer(args)
    instruct_args = deepcopy(args)
    instruct_args.gpt3_model = 'text-' + args.gpt3_model + '-001'
    instruct_model = load_summarizer(instruct_args)

    os.makedirs(os.path.join(args.save_dir, 'original'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'altered'), exist_ok=True)

    file_idx = len([x for x in os.listdir(os.path.join(args.save_dir, 'original'))]) # keep counting from wherever we left off, so we can resume progress as needed

    for fname in os.listdir(args.load_dir):
        path = os.path.join(args.load_dir, fname)
        if path in already_labeled_paths:
            continue

        with open(path, 'rb') as f:
            save_info = pickle.load(f)
        premise = save_info['premise']
        setting = save_info['setting']
        characters = save_info['characters']
        character_strings = save_info['character_strings']
        infer_attributes_string = save_info['infer_attributes_string']

        print('\n\n\n\nORIGINAL')
        print(infer_attributes_string)

        remaining_characters_to_sample = list(character_strings.keys())

        while True:
            if len(remaining_characters_to_sample) == 0:
                break
            resample_key = random.choice(remaining_characters_to_sample)
            remaining_characters_to_sample.remove(resample_key)
            context = infer_attributes_string.split('\n\n')
            prefix = ''
            for i, section in enumerate(context):
                if not section.startswith(resample_key):
                    prefix += section + '\n\n'
                else:
                    original_description = section[len(resample_key):]
                    suffix = '\n\n' + '\n\n'.join(context[i+1:])
                    break
            prefix += resample_key
            contradiction_entries = resample_description(prefix, suffix, resample_key, original_description, num_samples=5)

            should_break = False
            for entry in contradiction_entries:
                if entry['contradiction_logprob'] < -1:
                    print('REMAINING ENTRIES LOW LOGPROB; GO TO NEXT CHARACTER')
                    break

                print('\n\n')
                print('RESAMPLED CHARACTER')
                print(resample_key)
                print('ORIGINAL DESCRIPTION')
                print(resample_key + original_description)
                print('NEW DESCRIPTION')
                print(resample_key + entry['new_description'])
                print('PREDICTED CONTRADICTION ORIGINAL')
                print(entry['contradicted_original'])
                print('PREDICTED CONTRADICTION NEW')
                print(entry['contradictory_completion'])
                print('PREDICTED CONTRADICTION LOGPROB')
                print(entry['contradiction_logprob'])

                is_good = input('Is this description good? [y/n/s]')
                if is_good.startswith('n'):
                    continue
                if is_good.startswith('s'):
                    should_break = True
                    break

                new_characters = characters.replace(original_description, entry['new_description'])
                new_character_strings = deepcopy(character_strings)
                for entity in new_character_strings.values():
                    entity.reset_attributes()
                    entity.description = entity.description.replace(original_description, entry['new_description'])
                new_infer_attributes_string = infer_attributes_string.replace(original_description, entry['new_description'])
                should_break = True
                break
            if should_break:
                break

        with open(os.path.join(args.save_dir, 'already_labeled.txt'), 'a') as f:
            f.write(path + '\n')

        if not is_good.startswith('y'): # s for skip, n for no
            continue

        save_info['contradiction_logprob'] = entry['contradiction_logprob']
        save_info['contradicted_part'] = entry['contradicted_original']
        with open(os.path.join(args.save_dir, 'original', str(file_idx) + '.pkl'), 'wb') as wf:
            pickle.dump(save_info, wf)

        save_info['characters'] = new_characters
        save_info['character_strings'] = new_character_strings
        save_info['infer_attributes_string'] = new_infer_attributes_string
        save_info['contradicted_part'] = entry['contradictory_completion']
        with open(os.path.join(args.save_dir, 'altered', str(file_idx) + '.pkl'), 'wb') as wf:
            pickle.dump(save_info, wf)
        
        file_idx += 1