import argparse
import pickle
import os
from copy import deepcopy
import math
import random

from tqdm import trange
from sklearn.metrics import roc_auc_score

from story_generation.edit_module.entity import Entity, complete_mutual_relations
from story_generation.common.util import *
from story_generation.common.data.data_util import add_data_args, load_dataset
from story_generation.common.summarizer.summarizer_util import add_summarizer_args, load_summarizer


def evaluate_consistency(save_info_file, story_file, instruct_model, method='structured', reinfer=True, verbose=True, contradiction_threshold=0.5):
    assert method in ['entailment', 'entailment-dpr', 'structured'] # TODO add baselines
    if method == 'entailment':
        with open(story_file, 'r') as rf:
            story = rf.read()
            if story.strip() == 'SKIPPED':
                if verbose:
                    print('SKIPPED STORY')
                return None
        with open(save_info_file, 'rb') as f:
            save_info = pickle.load(f)
        base_info = save_info['infer_attributes_string']
        base_info_sentences = split_paragraphs(base_info, mode='sentence')
        story_sentences = split_paragraphs(story, mode='sentence')
        premises, hypotheses = [], []
        for s1 in base_info_sentences:
            for s2 in story_sentences:
                premises.append(s1)
                hypotheses.append(s2)
        logprobs, _ = score_entailment(premises, hypotheses)
        return np.exp(logprobs[:, 0]).max()
        # if np.exp(logprobs[:, 0]).max() > contradiction_threshold: # contradiction detected
        #     return 0
        # return 1
    elif method == 'entailment-dpr':
        with open(story_file, 'r') as rf:
            story = rf.read()
            if story.strip() == 'SKIPPED':
                if verbose:
                    print('SKIPPED STORY')
                return None
        with open(save_info_file, 'rb') as f:
            save_info = pickle.load(f)
        base_info = save_info['infer_attributes_string']
        base_info_sentences = split_paragraphs(base_info, mode='sentence')
        story_sentences = split_paragraphs(story, mode='sentence')
        premises, hypotheses = [], []
        for premise in story_sentences:
            scores = score_dpr(premise + 'Is this sentence consistent with the previous story?', base_info_sentences)
            premises.append(premise)
            hypotheses.append(base_info_sentences[scores.argmax()])
        logprobs, _ = score_entailment(premises, hypotheses)
        return np.exp(logprobs[:, 0]).max()
        # if np.exp(logprobs[:, 0]).max() > contradiction_threshold: # contradiction detected
        #     return 0
        # return 1
    else:
        # if the story file is SKIPPED, return None
        with open(story_file, 'r') as rf:
            story = rf.read()
            if story.strip() == 'SKIPPED':
                if verbose:
                    print('SKIPPED STORY')
                return None

        # infer attributes on the characters if it's not already there, and resave if necessary
        with open(save_info_file, 'rb') as f:
            save_info = pickle.load(f)
            for character in save_info['character_strings']:
                if type(save_info['character_strings'][character]) == dict: # for the data from the paper, we converted it to a dict when refactoring the code to avoid pkl reloading problems
                    char_info = save_info['character_strings'][character]
                    save_info['character_strings'][character] = Entity(char_info['name'], char_info['description'], char_info['is_character'], char_info['attributes'])
        
        story_detected_characters = deduplicate_match_entities(detect_entities(story, all_entities_dict=save_info['character_strings']), save_info['character_strings'].keys())[0]

        if all([len(save_info['character_strings'][character].attributes) == 0 for character in save_info['character_strings']]) or reinfer: # haven't inferred attributes yet, or want to reinfer
            if verbose:
                print('INFERRING INITIAL ATTRIBUTES')
            # for character in story_detected_characters: # no need to infer undetected characters since we wouldn't contradict against them anyway
            for character in save_info['character_strings'].keys():
                entity = save_info['character_strings'][character]
                entity.reset_attributes()
                entity.infer_attributes(save_info['infer_attributes_string'], instruct_model, other_names=[name for name in save_info['character_strings'].keys() if name != entity.name])
            complete_mutual_relations(save_info['character_strings'], instruct_model)
            with open(save_info_file, 'wb') as f:
                pickle.dump(save_info, f)
        
        if verbose:
            print('INFER ATTRIBUTES STRING')
            print(save_info['infer_attributes_string'])
            print('STORY')
            print(story)

        # infer attributes on the story file, only for the characters detected in this story passage
        contradiction_prob = 0
        for character in story_detected_characters:
            if verbose:
                print('CHARACTER')
                print(character)
            entity = save_info['character_strings'][character]
            if verbose:
                print('ATTRIBUTES BEFORE')
                print(entity.attributes)
            new_prob = entity.infer_attributes(story, instruct_model, other_names=[name for name in save_info['character_strings'].keys() if name != entity.name], return_contradiction_prob=True)
            contradiction_prob = max(contradiction_prob, new_prob)
            if verbose:
                print('ATTRIBUTES AFTER')
                print(entity.attributes)

        _, new_prob = complete_mutual_relations(save_info['character_strings'], instruct_model, return_contradiction_prob=True)
        return max(contradiction_prob, new_prob)
        # if len(mutual_relation_contradictions) > 0:
        #     print('CONTRADICTIONS')
        #     print(mutual_relation_contradictions)
        #     return 0
        # return 1


def evaluate_consistency_dataset(data_dir, instruct_model, method='structured', verbose=True, max_num_files=1000000, contradiction_threshold=0.5):
    num_files = len(os.listdir(os.path.join(data_dir, 'original_stories')))
    num_files = min(num_files, max_num_files)
    same_scores = []
    diff_scores = []

    for i in trange(0, num_files):
        for save_info_folder, story_folder in [('original', 'original_stories'), ('altered', 'altered_stories')]:

            if verbose:
                print('\n\n\n\nEXAMPLE ' + str(i))
                print('\n\nSAME PAIR:', save_info_folder, story_folder)
            score = evaluate_consistency(os.path.join(data_dir, save_info_folder, str(i) + '.pkl'), os.path.join(data_dir, story_folder, str(i) + '.txt'), instruct_model, method=method, verbose=verbose, contradiction_threshold=contradiction_threshold)
            if verbose:
                print('SCORE:', score)
                # print({1: 'CORRECT', 0: 'WRONG', None: 'N/A'}[score])
            if score is not None:
                same_scores.append(score)
        for save_info_folder, story_folder in [('original', 'altered_stories'), ('altered', 'original_stories')]:

            if verbose:
                print('\n\n\n\nEXAMPLE ' + str(i))
                print('\n\nDIFF PAIR:', save_info_folder, story_folder)
            score = evaluate_consistency(os.path.join(data_dir, save_info_folder, str(i) + '.pkl'), os.path.join(data_dir, story_folder, str(i) + '.txt'), instruct_model, method=method, verbose=verbose, contradiction_threshold=contradiction_threshold)
            if verbose:
                print('SCORE:', score)
                # print({0: 'CORRECT', 1: 'WRONG', None: 'N/A'}[score])
            if score is not None:
                diff_scores.append(score)
    assert len(same_scores) == len(diff_scores)
    print('ROC AUC', roc_auc_score([0 for _ in range(len(same_scores))] + [1 for _ in range(len(diff_scores))], same_scores + diff_scores))
    # if verbose:
    #     print('TOTAL', len(same_scores))
    #     print('SAME CONSISTENCY FRAC', sum(same_scores) / len(same_scores))
    #     print('DIFF CONSISTENCY FRAC', sum(diff_scores) / len(diff_scores))
    # return {'total': len(same_scores), 
    #         'same_frac': sum(same_scores) / len(same_scores), 
    #         'diff_frac': sum(diff_scores) / len(diff_scores), 
    #         'same_scores': same_scores,
    #         'diff_scores': diff_scores}


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser = add_general_args(parser)
    parser = add_summarizer_args(parser)
    parser.add_argument('--consistency-dataset-dir', type=str, required=True, help='dataset directory')
    parser.add_argument('--consistency-method', type=str, default='structured', choices=['structured', 'entailment', 'entailment-dpr'], help='consistency method')
    parser.add_argument('--contradiction-threshold', type=float, default=0.5, help='threshold for contradiction prob when using entailment baselines')
    args = parser.parse_args()

    base_model = load_summarizer(args)
    instruct_args = deepcopy(args)
    instruct_args.gpt3_model = 'text-' + args.gpt3_model + '-001'
    instruct_model = load_summarizer(instruct_args)

    results = evaluate_consistency_dataset(args.consistency_dataset_dir, instruct_model, method=args.consistency_method, verbose=not args.quiet, contradiction_threshold=args.contradiction_threshold)
