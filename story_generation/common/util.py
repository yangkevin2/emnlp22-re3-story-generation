import os
import sys
from contextlib import contextmanager
import time
import re
from collections import defaultdict
import string
import logging
from typing import *

import requests
import torch
from torch import Tensor
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForQuestionAnswering, T5TokenizerFast, T5ForConditionalGeneration
from flair.data import Sentence
from flair.models import SequenceTagger
import openai
from scipy.special import softmax
import numpy as np
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from Levenshtein import distance as levenshtein_distance
import signal

from story_generation.common.data.split_paragraphs import *

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

sentence_encoder = None
dpr_query_encoder = None
dpr_context_encoder = None
entailment_model = None
entailment_tokenizer = None
ner_model = None
qa_model = None
qa_tokenizer = None
coreference_model = None
gpt_tokenizer = None


def load_gpt_tokenizer():
    global gpt_tokenizer
    if gpt_tokenizer is None:
        gpt_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    return gpt_tokenizer


def load_sentence_encoder(device=('cuda' if torch.cuda.is_available() else 'cpu')):
    global sentence_encoder
    if sentence_encoder is None:
        logging.log(23, 'loading sentence model')
        sentence_encoder = SentenceTransformer('all-mpnet-base-v2')
        logging.log(23, 'done loading')
    return sentence_encoder


def load_dpr(device=('cuda' if torch.cuda.is_available() else 'cpu')):
    global dpr_query_encoder
    global dpr_context_encoder
    if dpr_context_encoder is None:
        logging.log(23, 'loading dpr model')
        dpr_query_encoder = SentenceTransformer('sentence-transformers/facebook-dpr-question_encoder-single-nq-base')
        dpr_context_encoder = SentenceTransformer('sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base')
        logging.log(23, 'loaded dpr model')
    return dpr_query_encoder.to(device), dpr_context_encoder.to(device)


def load_entailment_model(device=('cuda' if torch.cuda.is_available() else 'cpu')):
    global entailment_model
    global entailment_tokenizer
    if entailment_model is None:
        logging.log(23, 'loading entailment model')
        entailment_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
        entailment_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
        logging.log(23, 'loaded entailment model')
    return entailment_model.to(device), entailment_tokenizer


def load_ner_model(device=('cuda' if torch.cuda.is_available() else 'cpu')):
    global ner_model
    if ner_model is None:
        logging.log(23, 'loading ner model')
        ner_model = SequenceTagger.load("flair/pos-english")
        logging.log(23, 'loaded ner model')
    return ner_model.to(device)


def load_coreference_model(device=('cuda' if torch.cuda.is_available() else 'cpu')):
    global coreference_model
    if coreference_model is None:
        logging.log(23, 'loading coreference model')
        import spacy
        coreference_model = spacy.load('en_core_web_sm')
        import neuralcoref
        neuralcoref.add_to_pipe(coreference_model)
        logging.log(23, 'loaded coreference model')
    return coreference_model


def load_qa_model(device=('cuda' if torch.cuda.is_available() else 'cpu')):
    global qa_model
    global qa_tokenizer
    if qa_model is None:
        logging.log(23, 'loading qa model')
        # qa_model = pipeline('question-answering', model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2", device=0 if device=='cuda' else -1)
        model_name = "allenai/unifiedqa-t5-large"
        # model_name = "allenai/unifiedqa-t5-11b"
        qa_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        qa_tokenizer = T5TokenizerFast.from_pretrained(model_name)
        logging.log(23, 'loaded qa model')
    return qa_model, qa_tokenizer


@torch.no_grad()
def sentence_encode(sentences):
    sentence_encoder = load_sentence_encoder()
    return sentence_encoder.encode(sentences)


@torch.no_grad()
def sentence_similarity(query, contexts):
    query_encoded = sentence_encode([query])
    contexts_encoded = sentence_encode(contexts)
    return (query_encoded * contexts_encoded).sum(axis=1)


@torch.no_grad()
def entailment_preprocess(text):
    return text.replace("'s gender is", " is") # for some reason the entailment model is really bad at this construction


@torch.no_grad()
def score_entailment(premise, hypothesis):
    """
    Score entailment between two sentences
    """
    entailment_model, entailment_tokenizer = load_entailment_model()
    premise = [entailment_preprocess(premise)] if type(premise) == str else [entailment_preprocess(p) for p in premise]
    hypothesis = [entailment_preprocess(hypothesis)] if type(hypothesis) == str else [entailment_preprocess(h) for h in hypothesis]
    batch_inputs = entailment_tokenizer(premise, hypothesis, return_tensors='pt', padding=True)
    batch_inputs = {key:value.to(entailment_model.device) for key, value in batch_inputs.items()}
    logprobs = entailment_model(**batch_inputs).logits.log_softmax(dim=-1)
    consistent_logprobs = logprobs[:, 1:].logsumexp(dim=1) # log probability of not contradictory (contradiction is index 0)
    penalty = (-consistent_logprobs).max().item()
    return logprobs.cpu().numpy(), penalty


@torch.no_grad()
def get_agreed_facts(texts, threshold=0.5, agreement_threshold=1): # how many other texts have to entail this one before we accept it as true
    agreed_facts = []
    for text in texts:
        scores, _ = score_entailment(texts, [text for _ in range(len(texts))])
        scores = softmax(scores, axis=-1)
        if (scores[:, 2] > threshold).sum() -1 >= agreement_threshold: # don't include the text itself, since it should entail itself. check how many others it agrees with
            agreed_facts.append(text)
    return agreed_facts


@torch.no_grad()
def get_entailment_groups(texts_counts, threshold=0.5):
    # group texts based on one entailing the other, with logprob < threshold
    # specifically, return texts that aren't entailed by a text earlier in the list
    # the model here roughly is that there are several disconnected cliques in the graph and you want to return one from each
    # but actually some are more specific than others; when we detect this, replace the ones we selected earlier
    texts_list = list(texts_counts.keys())
    nonentailed_texts = defaultdict(lambda: 0)
    for i, text in enumerate(texts_list):
        if i == 0:
            nonentailed_texts[text] = texts_counts[text]
        else:
            scores, _ = score_entailment(texts_list[:i], [text for _ in range(i)])
            scores = softmax(scores, axis=-1) # probabilities for text to be entailed by the others
            keys = list(nonentailed_texts.keys())
            if scores[:, 2].max() < threshold:
                # now check if we should add this, or replace an existing one
                scores, _ = score_entailment([text for _ in range(len(keys))], keys)
                scores = softmax(scores, axis=-1)
                nonentailed_texts[text] = texts_counts[text]
                for j, t in enumerate(keys):
                    if scores[j, 2] >= threshold:
                        nonentailed_texts[text] += nonentailed_texts[t]
                        del nonentailed_texts[t]
                # nonentailed_texts = [t for j, t in enumerate(nonentailed_texts) if scores[j, 2] < threshold]
                # nonentailed_texts.append(text)
            else:
                for j, t in enumerate(keys):
                    if scores[j, 2] >= threshold:
                        nonentailed_texts[t] += texts_counts[text]
    return nonentailed_texts


@torch.no_grad()
def entailment_equals(premise, hypothesis, threshold=0.5):
    scores, _ = score_entailment(premise, hypothesis)
    scores = softmax(scores, axis=-1)
    return scores[:, 2].max() >= threshold


@torch.no_grad()
def consistent_equals(premise, hypothesis, threshold=0.5):
    scores, _ = score_entailment(premise, hypothesis)
    scores = softmax(scores, axis=-1)
    return scores[:, 2].max() + scores[:, 1].max() >= threshold


@torch.no_grad()
def score_qa(question, context, do_sample=False, num_beams=1, device=('cuda' if torch.cuda.is_available() else 'cpu')):
    qa_model, qa_tokenizer = load_qa_model()
    # qa_input = {'question': replace_newlines(question), 'context': replace_newlines(context)} # for some reason newlines screw it up
    # return qa_model(qa_input)
    input_string = question.strip() + ' \\n ' + context.strip()
    input_string = input_string.lower() # unifiedqa says to do this
    input_string = re.sub("'(.*)'", r"\1", input_string) # unifiedqa says to do this
    input_ids = qa_tokenizer.encode(input_string, return_tensors="pt").to(device)
    res = qa_model.generate(input_ids, do_sample=do_sample, num_beams=1 if do_sample else num_beams, num_return_sequences=num_beams, output_scores=True, early_stopping=True, return_dict_in_generate=True)
    return qa_tokenizer.batch_decode(res.sequences, skip_special_tokens=True), F.softmax(res.sequences_scores * res.sequences.shape[-1], dim=-1).cpu().tolist() if hasattr(res, 'sequences_scores') else [1./num_beams for _ in range(num_beams)]


def replace_tokens(text, replacement_dict):
    for token, replacement in replacement_dict.items():
        text = text.replace(token, replacement)
    return text


def split_list(text, prefix=None, strict=False):
    items = []
    list_idx = 1
    while str(list_idx) + '. ' in text:
        if not text.startswith(str(list_idx) + '. '):
            logging.log(23, 'Warning: bad list formatting')
            logging.log(23, text)
            return []
        text = text[len(str(list_idx) + '. '):]
        list_idx += 1
        pieces = text.split(str(list_idx) + '. ')
        if prefix is None or pieces[0].startswith(prefix):
            if '\n' in pieces[0].strip():
                if strict:
                    logging.log(23, 'Warning: bad list formatting')
                    logging.log(23, text)
                    return []
                else:
                    pieces[0] = pieces[0].strip().split('\n')[0].strip()
            items.append(pieces[0].strip())
        if len(pieces) > 1:
            text = str(list_idx) + '. ' + pieces[1]
    return items


def resolve_names(text, names, device=('cuda' if torch.cuda.is_available() else 'cpu')): # convert names to an unambiguous form, if detected. note this will destroy any newline formatting. 
    # logging.log(23, 'resolving names for text', text)
    if len(text.strip()) == 0:
        logging.log(23, 'Warning: detect entities on empty string')
        return text
    ner_model = load_ner_model(device=device)
    sentence = Sentence(text)
    ner_model.predict(sentence)
    spans = sentence.get_spans()
    name_spans = []
    for i, span in enumerate(spans):
        if span.tag in ['NNP', 'NNPS']:
            if i > 0 and spans[i-1].tag in ['NNP', 'NNPS']:
                name_spans[-1][0] += ' ' + span.text
            else:
                name_spans.append([span.text, True]) # named entities only here
        else:
            name_spans.append([span.text, False])
    for i in range(len(name_spans)):
        if name_spans[i][1]: # it's a name, possibly multiple spans
            ent = name_spans[i][0]
            matched_entities = []
            for prior_entity in names:
                if ent in prior_entity or prior_entity in ent:
                    matched_entities.append(prior_entity)
                elif ent.split()[0] == prior_entity.split()[0]: # avoid edge cases with changing last names
                    matched_entities.append(prior_entity)
            if len(matched_entities) == 1: # unambiguously matches exactly 1 character
                name_spans[i][0] = matched_entities[0]
    resolved_text = ''
    for span in name_spans:
        if span[0][0] in string.punctuation:
            resolved_text += span[0]
        else:
            resolved_text += ' ' + span[0]
    resolved_text = resolved_text.strip()
    # logging.log(23, 'after resolving:', resolved_text)
    return resolved_text


def gpt3_edit(text, instruction, prefix=None, filter_append=True, temperature=0.7, top_p=1, num_completions=5, num_iters=3, max_retries=3, **kwargs):
    for _ in range(num_iters): # sometimes it fails to actually make the proper change, but we have some checks in place to make sure it doesn't do bad things most of the time, so try multiple times
        retry = True
        retry_num = 0
        while retry:
            retry_num += 1
            if retry_num > max_retries:
                logging.log(23, 'Warning: gpt3 edit failed to make a change after ' + str(max_retries) + ' attempts')
                return text
            try:
                with time_limit(30):
                    completion = openai.Edit.create(
                                        engine='text-davinci-edit-001',
                                        input=text if prefix is None else prefix.strip() + ' ' + text,
                                        instruction=instruction,
                                        temperature=temperature,
                                        top_p=top_p,
                                        n=num_completions,
                                        **kwargs
                    )
                if all(['text' in completion['choices'][i] for i in range(num_completions)]):
                    retry = False
                else:
                    raise ValueError
            except Exception as e: 
                logging.log(23, str(e))
                time.sleep(0.2)
                logging.log(23, 'retrying...')
        tokenizer = load_gpt_tokenizer()
        outputs = [completion['choices'][i]['text'] for i in range(num_completions)]
        logging.log(21, 'GPT3 CALL'  + ' ' + 'text-davinci-edit-001' + ' ' + str(len(tokenizer.encode(text if prefix is None else prefix.strip() + ' ' + text)) + sum([len(tokenizer.encode(o)) for o in outputs])))
        edited_texts = []
        for i in range(num_completions):
            edited_text = completion['choices'][i]['text']
            context_consistency_score = 0
            if filter_append: # sometimes the edit engine just appends text to the end, which isn't what we want. try to detect and remove this. 
                done_filter_append = False
                for i in [50, 40, 30, 20]:
                    text_suffix = text.strip()[-i:] # if the last 50 chars are repeated verbatim, cut everything afterward
                    if text_suffix in edited_text and edited_text.count(text_suffix) == 1:
                        edited_text = edited_text.split(text_suffix)[0] + text_suffix
                        done_filter_append = True
                        break
                if not done_filter_append:
                    # try to detect sentences at the end of the edited text which contain a lot of repetition with the instruction. repeat until this isn't the case for the last sentence.
                    for i in range(100): # shouldn't ever hit 100 except in weird edge cases
                        if len(edited_text.strip()) == 0:
                            break
                        last_sentence = split_paragraphs(edited_text, mode='sentence')[-1].lower()
                        instruction_words = instruction.lower().split(' ')
                        modified = False
                        for i in range(6, len(instruction_words)):
                            if ' '.join(instruction_words[i-5:i]) in last_sentence:
                                edited_text = cut_last_sentence(edited_text)
                                modified = True
                                break
                        if not modified:
                            break
            if prefix is not None:
                done_processing = False
                for i in [50, 40, 30, 20]:
                    prefix_suffix = prefix.strip()[-i:]
                    if prefix_suffix in edited_text:
                        split_text = edited_text.split(prefix_suffix)
                        if len(split_text) == 2:
                            edited_text = split_text[1]
                            edited_context = split_text[0] + prefix_suffix
                            done_processing = True
                            break
                if not done_processing:
                    for i in [50, 40, 30, 20]:
                        text_prefix = text.strip()[:i]
                        if text_prefix in edited_text:
                            split_text = edited_text.split(text_prefix)
                            if len(split_text) == 2:
                                edited_text = text_prefix + split_text[1]
                                edited_context = split_text[0]
                                done_processing = True
                                break
                if not done_processing:
                    logging.log(23, 'Warning: could not remove prefix context when editing')
                    logging.log(23, 'PREFIX' + ' ' + prefix)
                    logging.log(23, 'TEXT' + ' ' + text)
                    logging.log(23, 'EDITED TEXT' + ' ' + edited_text)
                    logging.log(23, 'returning original unedited text')
                    edited_text = text
                    context_consistency_score = -1000
                else:
                    edited_text = edited_text.lstrip()
                    context_consistency_score = -detect_num_changed_names(prefix, edited_context)
                    # if text != edited_text:
                    #     context_consistency_score += 0.5 # bonus for making some change
                    context_consistency_score += 0.001 * min(50, levenshtein_distance(text, edited_text)) # small bonus for making some kind of edit to the original text
            edited_texts.append((edited_text, context_consistency_score))
        edited_texts = sorted(edited_texts, key=lambda x: x[1], reverse=True)

        logging.log(23, '\n\nEDITED' + ' ' + edited_texts[0][0])
        if text != edited_texts[0][0]:
            text = edited_texts[0][0]
            break
        text = edited_texts[0][0] # the one that stays closest to the original autoregressive context, if provided. so not changing things that were already decided previously.
    return text


def gpt3_insert(prefix, suffix, top_p=1, temperature=1, max_tokens=256, frequency_penalty=0, presence_penalty=0, **kwargs):
    retry = True
    while retry:
        try:
            completion = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prefix,
                suffix=suffix,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                **kwargs
            )
            retry = False
        except Exception as e: 
            logging.log(23, e)
            time.sleep(0.2)
            logging.log(23, 'retrying...')
    outputs = [completion['choices'][i]['text'] for i in range(len(completion['choices']))]
    tokenizer = load_gpt_tokenizer()
    logging.log(21, 'GPT3 CALL' + ' ' + 'text-davinci-002' + ' ' + str(len(tokenizer.encode(prefix)) + len(tokenizer.encode(suffix)) + sum([len(tokenizer.encode(o)) for o in outputs])))
    return outputs


def detect_num_changed_names(context, edited_context): # rough heuristic for how many names were changed in the context, to get a sense of how off the editing model was for reranking
    return levenshtein_array_distance(detect_entities(context), detect_entities(edited_context))


def levenshtein_array_distance(array1, array2):
    decoding = list(set(array1+array2))
    encoding = {v:k for k,v in enumerate(decoding)}
    encoded_array1 = ''.join([chr(encoding[x]) for x in array1])
    encoded_array2 = ''.join([chr(encoding[x]) for x in array2])
    return levenshtein_distance(encoded_array1, encoded_array2)


def resample_description(prefix, suffix, name, original_description, num_samples=1):
    tokenizer = load_gpt_tokenizer()
    avoid_words = [word for word in original_description.split() if word not in prefix and word not in suffix]
    logit_bias = get_repetition_logit_bias(tokenizer, ' '.join(avoid_words), -10, include_upper=True)
    # tokens = [tok for tok in set(tokenizer.encode(original_description)) if tok not in set(tokenizer.encode(prefix + suffix))]
    # logit_bias = {tok:-100 for tok in tokens}
    completions = gpt3_insert(prefix, suffix, logit_bias=logit_bias, stop='\n', n=num_samples)
    completions_split = [split_paragraphs(name + completion, mode='sentence') for completion in completions]
    completion_sentences = sum(completions_split, [])
    sentence_idx_to_original_completion_idx = []
    for i, split in enumerate(completions_split):
        for _ in split:
            sentence_idx_to_original_completion_idx.append(i)
    contradiction_entries = []
    split_original_description = split_paragraphs(name + original_description, mode='sentence')
    for original_sentence in split_original_description:
        entailment_scores = score_entailment([original_sentence for _ in range(len(completion_sentences))], completion_sentences)[0]
        for i in range(len(entailment_scores)):
            contradiction_entries.append({
                'contradicted_original': original_sentence,
                'contradictory_completion': completion_sentences[i],
                'contradiction_logprob': entailment_scores[i, 0],
                'new_description': completions[sentence_idx_to_original_completion_idx[i]],
            })
    contradiction_entries = sorted(contradiction_entries, key=lambda x: x['contradiction_logprob'], reverse=True)
    return contradiction_entries


def replace_coreferences(text):
    coreference_model = load_coreference_model()
    doc = coreference_model(text)
    return doc._.coref_resolved


def get_common_tokens(tokenizer):
    sw = [w.lower() for w in stopwords.words('english')]
    token_string = ''
    for word in sw:
        token_string += ' ' + word
        token_string += ' ' + word[0].upper() + word[1:]
    token_string += string.punctuation
    return set(tokenizer.encode(token_string))


def get_repetition_logit_bias(tokenizer, text, bias, bias_common_tokens=False, existing_logit_bias=None, include_upper=False):
    logit_bias = {} if existing_logit_bias is None else existing_logit_bias
    for word in text.strip().split():
        processed_word = word.strip().lower()
        tokens = tokenizer.encode(word.strip()) + \
                 tokenizer.encode(' ' + word.strip())
                #  tokenizer.encode(processed_word) + \
                #  tokenizer.encode(' ' + processed_word)
                #  tokenizer.encode(processed_word[0].upper() + processed_word[1:]) + \
                #  tokenizer.encode(' ' + processed_word[0].upper() + processed_word[1:])
                #  tokenizer.encode(processed_word.upper()) + \
                #  tokenizer.encode(' ' + processed_word.upper()) + \
        if include_upper:
            tokens += tokenizer.encode(processed_word.upper()) + \
                      tokenizer.encode(' ' + processed_word.upper())
        for tok in set(tokens):
            logit_bias[tok] = bias
    if not bias_common_tokens: # don't bias against common tokens (stopwords + punc)
        for tok in get_common_tokens(tokenizer):
            if tok in logit_bias:
                del logit_bias[tok]
    return logit_bias


def strip_shared_names(text, full_names):
    # strip the text of shared parts of full names which screw up the coreference system, e.g. shared last names
    names_to_strip = set()
    name_components = set()
    for name in full_names:
        for component in name.strip().split():
            if component in name_components:
                names_to_strip.add(component)
            name_components.add(component)
    for name in full_names:
        full_name = name
        for component in names_to_strip:
            if component in name:
                name = name.replace(component, '')
        while '  ' in name:
            name = name.replace('  ', ' ')
        name = name.strip()
        text = text.replace(full_name, name)
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text


@torch.no_grad()
def pos_tag(text, device=('cuda' if torch.cuda.is_available() else 'cpu')):
    ner_model = load_ner_model(device=device)
    sentence = Sentence(text)
    ner_model.predict(sentence)
    spans = sentence.get_spans()
    return spans


@torch.no_grad()
def detect_entities(text, add_dpr_entities=False, all_entities_dict=None, include_unnamed=False, device=('cuda' if torch.cuda.is_available() else 'cpu')):
    if len(text.strip()) == 0:
        logging.log(23, 'Warning: detect entities on empty string')
        return []
    matched_entities = []
    ner_model = load_ner_model(device=device)
    for text_section in split_paragraphs(text, mode='newline'):
        sentence = Sentence(text_section)
        ner_model.predict(sentence)
        spans = sentence.get_spans()
        for i, span in enumerate(spans):
            if span.tag in ['NNP', 'NNPS']:
                if i > 0 and spans[i-1].tag in ['NNP', 'NNPS']:
                    matched_entities[-1] += ' ' + span.text
                else:
                    matched_entities.append(span.text) # named entities only here
            elif include_unnamed and span.tag in ['NN', 'NNS']:
                matched_entities.append(span.text)
    for i in range(len(matched_entities)):
        matched_entities[i] = matched_entities[i].strip().strip(string.punctuation)

    # ner_model, ner_tokenizer = load_ner_model(device=device)
    # nlp = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, device=0 if device == 'cuda' else -1)
    # entity_tokens = nlp(text)
    # entities = []
    # previous_end = -1
    # for token_info in entity_tokens: # TODO could make sure it ends the word e.g. it detects Astri when it's supposed to be Astrid
    #     if len(entities) > 0 and token_info['start'] == previous_end+1: # continuation of previous entity
    #         entities[-1] = entities[-1] + ' ' + token_info['word'].strip()
    #     elif len(entities) > 0 and token_info['start'] == previous_end:
    #         entities[-1] = entities[-1] + token_info['word'].strip('#')
    #     else:
    #         entities.append(token_info['word'])
    #     previous_end = token_info['end']
    # matched_entities = list(set(entities))
    
    if add_dpr_entities:
        # add any dpr retrieved entities that have a score higher than any of the name-matched entities
        dpr_query_encoder, dpr_context_encoder = load_dpr()
        keys, queries = [], []
        for k, v in all_entities_dict.items():
            if k not in ['premise', 'setting']:
                keys.append(k)
                queries.append(v.description + '\n\nFind additional information about ' + k + '.')
        query_encodings = dpr_query_encoder.encode(queries)
        context_encodings = dpr_context_encoder.encode(text)
        dpr_scores = (query_encodings * context_encodings.reshape(1, -1)).sum(axis=1)
        if any([ent in keys for ent in matched_entities]):
            min_matched_entity_score = min([dpr_scores[keys.index(ent)] for ent in matched_entities if ent in keys])
            additional_dpr_entities = []
            for k, s in zip(keys, dpr_scores):
                if k not in matched_entities and s > min_matched_entity_score:
                    additional_dpr_entities.append(k)
            matched_entities += additional_dpr_entities
        else:
            matched_entities += [keys[dpr_scores.argmax()]] # assume at least one entity shows up
            additional_dpr_entities = [keys[dpr_scores.argmax()]]
    return matched_entities


def deduplicate_match_entities(entities, names):
    # try to figure out which entities are the same as existing entities in our dict, and which are new; also deduplicate
    # TODO this is a bit hacky, but it's a start
    entities = [ent for ent in entities if ent not in ['Premise', 'Setting'] and len(ent.strip()) > 0] # special entities that we don't need to match
    entities = sorted([ent for ent in list(set(entities)) if ent[0].isupper()], key=lambda x: len(x), reverse=True)
    matched_entities, new_entities = set(), set()
    replacements = {}
    for ent in entities:
        if ent in new_entities:
            continue
        matched = False
        for prior_entity in names:
            if ent in prior_entity or prior_entity in ent:
                matched_entities.add(prior_entity)
                matched = True
                break
            elif ent.split()[0] == prior_entity.split()[0]: # avoid edge cases with changing last names
                logging.log(23, 'MATCHING EDGE CASE: ' + ent + ' ' + prior_entity)
                matched_entities.add(prior_entity)
                replacements[ent] = prior_entity
                matched = True
                break
        if not matched:
            for new_entity in new_entities:
                if ent in new_entity or new_entity in ent:
                    matched = True
                    break
        if not matched:
            new_entities.add(ent)
    return matched_entities, new_entities, replacements


def score_dpr(query, keys):
    dpr_query_encoder, dpr_context_encoder = load_dpr()
    query_encoding = dpr_query_encoder.encode(query)
    context_encoding = dpr_context_encoder.encode(keys)
    scores = (query_encoding.reshape(1, -1) * context_encoding).sum(axis=1)
    return scores


def replace_newlines(text):
    while '\n\n' in text:
        text = text.replace('\n\n', '\n')
    return text.replace('\n', ' ')


def add_general_args(parser):
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--seed', type=int, default=12345, help='seed')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--quiet', action='store_true', help='quiet mode')
    return parser


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def pad_to_max_length(tensors, dim, value=0):
    """
    Pad all tensors in the list to the max length along the given dim using the given value
    """
    max_length = max([tensor.shape[dim] for tensor in tensors])
    return [pad_to_length(tensor, max_length, dim, value=value) for tensor in tensors]


def pad_to_length(tensor, length, dim, value=0):
    """
    Pad tensor to given length in given dim using given value (value should be numeric)
    """
    assert tensor.size(dim) <= length
    if tensor.size(dim) < length:
        zeros_shape = list(tensor.shape)
        zeros_shape[dim] = length - tensor.size(dim)
        zeros_shape = tuple(zeros_shape)
        return torch.cat([tensor, torch.zeros(zeros_shape).type(tensor.type()).to(tensor.device).fill_(value)], dim=dim)
    else:
        return tensor


def pad_mask(lengths: torch.LongTensor) -> torch.ByteTensor:
    """
    Create a mask of seq x batch where seq = max(lengths), with 0 in padding locations and 1 otherwise. 
    """
    # lengths: bs. Ex: [2, 3, 1]
    lengths = torch.LongTensor(lengths)
    max_seqlen = torch.max(lengths)
    expanded_lengths = lengths.unsqueeze(0).repeat((max_seqlen, 1))  # [[2, 3, 1], [2, 3, 1], [2, 3, 1]]
    indices = torch.arange(max_seqlen).unsqueeze(1).repeat((1, lengths.size(0))).to(lengths.device)  # [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

    return expanded_lengths > indices  # pad locations are 0. #[[1, 1, 1], [1, 1, 0], [0, 1, 0]]. seqlen x bs


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    indices_to_keep = logits > -1e8
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        indices_to_keep = logits >= torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        sorted_indices_to_keep = ~sorted_indices_to_remove

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        indices_to_keep = sorted_indices_to_keep.scatter(1, sorted_indices, sorted_indices_to_keep)
        logits[indices_to_remove] = filter_value
    return logits, indices_to_keep


class ProgressMeter(object):
    """
    Display meter
    """
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries.append(time.ctime(time.time()))
        entries += [str(meter) for meter in self.meters]
        logging.log(23, '\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """
    Display meter
    Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if 'torch' in str(type(val)):
            val = val.detach()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


# https://github.com/alpa-projects/alpa/blob/main/examples/opt_serving/client.py
class AlpaOPTClient(object):
    
    def __init__(self,
                 url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 default_model: str = "default") -> None:
        if url is None:
            url = "https://opt.alpa.ai" # public api

        self.default_model = default_model
        self.completions_url = url + "/completions"
        self.logprobs_url = url + "/logprobs"
        self.api_key = api_key

    def refresh_url(self, alpa_url, alpa_port):
        if alpa_url.startswith('http'):
            self.completions_url = alpa_url + "/completions"
            self.logprobs_url = alpa_url + "/logprobs"
        else:
            logging.log(22, 'refreshing alpa url with file: ' + alpa_url)
            with open(alpa_url, 'r') as rf:
                alpa_hostname = rf.read().strip().split()[0]
                alpa_url = f'http://{alpa_hostname}:{alpa_port}'
                self.completions_url = alpa_url + "/completions"
                self.logprobs_url = alpa_url + "/logprobs"

    def completions(
        self,
        prompt: Union[str, Sequence[str], Sequence[int], Sequence[Sequence[int]]],
        min_tokens: int = 0,
        max_tokens: int = 32,
        top_p: float = 1.0,
        temperature: float = 1.0,
        echo: bool = True,
        model: Optional[str] = None,
    ) -> Dict:
        """
        Generation API.
        Parameters match those of the OpenAI API.
        https://beta.openai.com/docs/api-reference/completions/create

        Args:
          prompt: a list of tokenized inputs.
          min_tokens: The minimum number of tokens to generate.
          max_tokens: The maximum number of tokens to generate.
          temperature: What sampling temperature to use.
          top_p: The nucleus sampling probability.
          echo: if true, returned text/tokens/scores includes the prompt.
        """
        pload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "echo": echo,
            "api_key": self.api_key
        }
        result = requests.post(self.completions_url, json=pload, headers={"User-Agent": "Alpa Client"})
        return self.result_or_error(result)

    def logprobs(
        self,
        prompt: Union[str, Sequence[str], Sequence[int], Sequence[Sequence[int]]],
        top_p: float = 1,
        top_k: int = 100,
        cache_id: Optional = None,
        model: Optional[str] = None) -> Dict:
        """Return the log probability of the next top-k tokens"""
        pload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "top_p": top_p,
            "top_k": top_k,
            "api_key": self.api_key
        }
        if cache_id:
            pload["cache_id"] = cache_id
        result = requests.post(self.logprobs_url, json=pload, headers={"User-Agent": "Alpa Client"})
        return self.result_or_error(result)

    def result_or_error(self, result):
        result = result.json()
        if result.get("type", "") == "error":
            raise RuntimeError(
                result["stacktrace"] +
                f'RuntimeError("{result["message"]}")')
        else:
            return result