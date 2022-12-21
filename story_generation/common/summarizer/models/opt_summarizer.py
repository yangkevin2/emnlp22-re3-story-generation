from re import T
import time
import logging
import math
from copy import deepcopy

from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer
from scipy.special import expm1

from story_generation.common.summarizer.models.abstract_summarizer import AbstractSummarizer
from story_generation.common.data.split_paragraphs import split_paragraphs, cut_last_sentence
from story_generation.common.util import *

OPT_LOGPROBS_MAX_BS = 4

# tokenizer ids for tokens which contain " or ' which causes some problematic punc/spacing sometimes. prefer to use e.g. “ and ” instead in stories.
OPT_MACHINE_QUOTE_IDS = [22, 60, 72, 113, 845, 1297, 1917, 2901, 4332, 4805, 6697, 7862, 8070, 9957, 10076, 11227, 13198, 14025, 14220, 16844, 17495, 17523, 18456, 18653, 19207, 19651, 22896, 23962, 24095, 24337, 24464, 24681, 24992, 25718, 27223, 28553, 28578, 30550, 30697, 30831, 31051, 33525, 34133, 35290, 35347, 36856, 37008, 37637, 39058, 39732, 40021, 40389, 40635, 41039, 41066, 41758, 42078, 42248, 42255, 42777, 43012, 43074, 43101, 43775, 43809, 44065, 44374, 44690, 44717, 44757, 44926, 45333, 45390, 45751, 45863, 45894, 46150, 46294, 46353, 46469, 46479, 46481, 46671, 46679, 47096, 47460, 47770, 47919, 48110, 48149, 48298, 48336, 48474, 48615, 48742, 48789, 48805, 48880, 48893, 49070, 49177, 49189, 49293, 49329, 49434, 49509, 49608, 49643, 49667, 49713, 49721, 49738, 49761, 49778, 49784, 49799, 49817, 49849, 49852, 49853, 49871, 49900, 49923, 49991, 49995, 50000, 50003, 50020, 50154, 50184, 50206] + [18, 75, 108, 128, 214, 348, 437, 581, 955, 1017, 1598, 2652, 3934, 6600, 9376, 9957, 10076, 10559, 12801, 13373, 13864, 17809, 19651, 22896, 23500, 24095, 24464, 24992, 27144, 27645, 30171, 31509, 32269, 35347, 35661, 41667, 41734, 41833, 44162, 44294, 44403, 45393, 45803, 46117, 46150, 46250, 46495, 47033, 47052, 47429, 47579, 48694, 48759, 48817, 49201, 49515, 49525, 49690, 49836, 49888]

class OPTSummarizer(AbstractSummarizer):
    def __init__(self, args):
        assert args.alpa_url is not None
        if args.alpa_url.startswith('http'):
            alpa_url = args.alpa_url
        else:
            with open(args.alpa_url, 'r') as rf:
                alpa_hostname = rf.read().strip().split()[0]
                alpa_url = f'http://{alpa_hostname}:{args.alpa_port}'
        self.client = AlpaOPTClient(url=alpa_url, api_key=args.alpa_key)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
        self.tokenizer.add_bos_token = False
        self.args = args
        self.controller = None
    
    @torch.no_grad()
    def generate(self,
                 prompt,
                 generation_max_length=None,
                 top_p=None, 
                 temperature=None, 
                 retry_until_success=True, 
                 verbose=False, 
                 stop=None, 
                 num_completions=1, 
                 cut_sentence=False):
        assert type(prompt) == str
        logging.log(21, 'OPT GENERATION PROMPT')
        logging.log(21, prompt)
        if generation_max_length is None:
            generation_max_length = self.args.generation_max_length
        if top_p is None:
            top_p = self.args.summarizer_top_p
        if temperature is None:
            temperature = self.args.summarizer_temperature
        if stop is None:
            stop = []
        if type(stop) == str:
            stop = [stop]
        retry = True
        num_fails = 0
        while retry:
            try:
                completions = self.client.completions([prompt for _ in range(num_completions)], temperature=temperature, top_p=top_p, max_tokens=generation_max_length)
                if 'choices' not in completions:
                    import pdb; pdb.set_trace()
                completions = [entry['text'][len(prompt):] for entry in completions['choices']]
                retry = False
            except Exception as e:
                logging.warning(str(e))
                retry = retry_until_success
                num_fails += 1
                if retry:
                    logging.warning('retrying...')
                    time.sleep(num_fails)
                    logging.warning('old alpa url: ' + self.client.logprobs_url + ' at time ' + str(time.ctime()))
                    self.client.refresh_url(self.args.alpa_url, self.args.alpa_port)
                    logging.warning('new alpa url: ' + self.client.logprobs_url)
        for i, text in enumerate(completions):
            for s in stop:
                if s in text:
                    text = text[:text.index(s)]
            completions[i] = text
        if cut_sentence: 
            completions = [cut_last_sentence(text) for text in completions]
        return completions

    @torch.no_grad()
    def __call__(self, 
                 texts, 
                 controllers=None,
                 controller_initial_texts=None,
                 control_strengths=None,
                 generation_max_length=None,
                 top_p=1, 
                 top_k=100,
                 temperature=None, 
                 retry_until_success=True, 
                 verbose=False, 
                 stop=None, 
                 logit_bias=None, 
                 exclude_strings=None,
                 num_completions=1,
                 frequency_penalty=None,
                 presence_penalty=None,
                 cut_sentence=False,
                 bias_machine_quotes=True,
                 logit_bias_decay=1):
        assert type(texts) == list
        if logit_bias is None:
            logit_bias = {}
        assert controller_initial_texts is not None
        assert type(controller_initial_texts) == list and len(controllers) == len(control_strengths) and len(controller_initial_texts) == len(texts)
        if generation_max_length is None:
            generation_max_length = self.args.generation_max_length
        if top_p is None:
            top_p = self.args.summarizer_top_p
        if top_k is None:
            top_k = self.args.summarizer_top_k
        if temperature is None:
            temperature = self.args.summarizer_temperature
        if frequency_penalty is None:
            frequency_penalty = self.args.summarizer_frequency_penalty
        if presence_penalty is None:
            presence_penalty = self.args.summarizer_presence_penalty
        if stop is None:
            stop = []
        if type(stop) == str:
            stop = [stop]
        exclude_tokens = set()
        if exclude_strings is not None:
            for s in exclude_strings:
                exclude_tokens.update(self.tokenizer.encode(s[0].upper() + s[1:]))
                exclude_tokens.update(self.tokenizer.encode(' ' + s[0].upper() + s[1:]))
                exclude_tokens.update(self.tokenizer.encode(s[0].lower() + s[1:]))
                exclude_tokens.update(self.tokenizer.encode(' ' + s[0].lower() + s[1:]))
        sentences = []
        for text_idx, text in enumerate(texts):
            context_length = len(self.tokenizer.encode(text))
            if context_length > self.args.max_context_length - generation_max_length:
                logging.warning('context length' + ' ' + str(context_length) + ' ' + 'exceeded artificial context length limit' + ' ' + str(self.args.max_context_length - generation_max_length))
                # time.sleep(5) # similar interface to gpt3 query failing and retrying
                print('TOO LONG CONTEXT: ' + text)
                print('CONTEXT LENGTH:' + str(context_length))
            current_controller_initial_texts = controller_initial_texts[text_idx]
            assert len(current_controller_initial_texts) == len(controllers)
            logging.log(21, 'OPT CALL PROMPT')
            logging.log(21, text)
            device = controllers[0].device if (controllers is not None and len(controllers) > 0) else ('cuda' if torch.cuda.is_available() else 'cpu')
            expanded_logit_bias = torch.zeros(num_completions, self.tokenizer.vocab_size + 10).to(device)
            for token, bias in logit_bias.items():
                if token not in exclude_tokens:
                    expanded_logit_bias[:, token] = bias
            if bias_machine_quotes:
                for token in OPT_MACHINE_QUOTE_IDS:
                    expanded_logit_bias[:, token] = -100
            frequency_bias = torch.zeros_like(expanded_logit_bias)
            prompt = [[int(x) for x in self.tokenizer.encode(text)] for _ in range(num_completions)]
            if controllers is not None:
                controller_ids = []
                for ci in range(len(controllers)):
                    controller_ids.append([[int(x) for x in self.tokenizer.encode(current_controller_initial_texts[ci])] for _ in range(num_completions)])
            initial_prompt_length = len(prompt[0])
            cache_id = None
            for _ in range(generation_max_length):
                retry = True
                num_fails = 0
                while retry:
                    try:
                        with time_limit(30):
                            output = self.client.logprobs(prompt, top_p=top_p, top_k=top_k, cache_id=cache_id)
                        assert 'indices' in output and 'logprobs' in output
                        retry = False
                    except Exception as e:
                        logging.warning(str(e))
                        cache_id = None # not reentrant; restart cache
                        retry = retry_until_success
                        num_fails += 1
                        if retry:
                            logging.warning('retrying...')
                            time.sleep(num_fails)
                            logging.warning('old alpa url: ' + self.client.logprobs_url + ' at time ' + str(time.ctime()))
                            self.client.refresh_url(self.args.alpa_url, self.args.alpa_port)
                            logging.warning('new alpa url: ' + self.client.logprobs_url)
                distribution = (torch.zeros(num_completions, self.tokenizer.vocab_size + 10) - 1e8).to(device)
                distribution.scatter_(1, torch.LongTensor(output['indices']).to(device), torch.Tensor(output['logprobs']).to(device))
                if controllers is not None:
                    """
                    lm_logits: beam x 1 x vocab
                    input_ids: beam x seqlen
                    optionally, top_logits and top_indices, both beam x 1 x topk
                    """
                    for ci in range(len(controllers)):
                        # this call modifies and returns the distribution based on the given control string
                        distribution = controllers[ci](distribution.view(num_completions, 1, -1).to(device), 
                                                torch.LongTensor(controller_ids[ci]).view(num_completions, -1).to(device), 
                                                top_logits=torch.Tensor(output['logprobs']).view(num_completions, 1, -1).to(device), 
                                                top_indices=torch.LongTensor(output['indices']).view(num_completions, 1, -1).to(device),
                                                control_strength=control_strengths[ci])
                    distribution = distribution.squeeze(1)
                distribution /= temperature
                distribution += expanded_logit_bias + frequency_bias
                distribution = torch.softmax(distribution, dim=1)
                next_tokens = torch.multinomial(distribution, 1).squeeze(1)
                for i in range(num_completions):
                    prompt[i].append(next_tokens[i].item())
                    if controllers is not None:
                        for ci in range(len(controllers)):
                            controller_ids[ci][i].append(next_tokens[i].item())
                    if next_tokens[i].item() not in exclude_tokens:
                        frequency_bias[i, next_tokens[i].item()] -= frequency_penalty
                        if frequency_bias[i, next_tokens[i].item()] > -presence_penalty:
                            frequency_bias[i, next_tokens[i].item()] -= presence_penalty
                frequency_bias = frequency_bias * logit_bias_decay
                cache_id = output["cache_id"]
            for completion in prompt:
                decoded_completion = self.tokenizer.decode(completion[initial_prompt_length:])
                sentences.append(decoded_completion)

        for i in range(len(sentences)):
            sentence = sentences[i]
            while len(sentence) > 0 and sentence[-1] not in string.printable: # sometimes you get half of a special char at the end
                sentence = sentence[:-1]
            if len(self.tokenizer.encode(sentence.split('\n')[-1])) < 10: # if we just barely started a new paragraph, don't include it; you can get led down bad paths
                sentence = '\n'.join(sentence.split('\n')[:-1]).rstrip()
            sentence = sentence.rstrip()
            for s in stop:
                if s in sentence:
                    sentence = sentence[:sentence.index(s)]
            sentences[i] = sentence.rstrip()
        if cut_sentence:
            sentences = [cut_last_sentence(sentence) for sentence in sentences]
        return sentences
    
    @torch.no_grad()
    def generate_with_controller(self, 
                                 controllers, 
                                 controller_initial_texts,
                                 prompt, 
                                 control_strengths=None,
                                 generation_max_length=1,
                                 num_completions=1,
                                 fudge_top_k=100, 
                                 fudge_top_p=1, 
                                 temperature=None,
                                 logit_bias=None,
                                 exclude_strings=None,
                                 cut_sentence=False,
                                 logit_bias_decay=1):
        if logit_bias is None:
            logit_bias = {}
        completions = []
        for i in range(0, num_completions, OPT_LOGPROBS_MAX_BS):
            completions +=  self([prompt], 
                        controllers=controllers, 
                        control_strengths=control_strengths,
                        controller_initial_texts=[controller_initial_texts], 
                        generation_max_length=generation_max_length,
                        top_p=fudge_top_p,
                        top_k=fudge_top_k,
                        temperature=temperature,
                        logit_bias=logit_bias,
                        exclude_strings=exclude_strings,
                        num_completions=min(num_completions - i, OPT_LOGPROBS_MAX_BS),
                        cut_sentence=cut_sentence,
                        logit_bias_decay=logit_bias_decay)
        return completions
    
    def create_logit_bias_for_prompt(self, prompt, bias=0, exclude_strings=None, logit_bias=None, use_frequency_penalty=False, decay=1):
        if logit_bias is None:
            logit_bias = {}
        exclude_tokens = set()
        if exclude_strings is not None:
            for s in exclude_strings:
                exclude_tokens.update(self.tokenizer.encode(s[0].upper() + s[1:]))
                exclude_tokens.update(self.tokenizer.encode(' ' + s[0].upper() + s[1:]))
                exclude_tokens.update(self.tokenizer.encode(s[0].lower() + s[1:]))
                exclude_tokens.update(self.tokenizer.encode(' ' + s[0].lower() + s[1:]))
        for i, token in enumerate(reversed(self.tokenizer.encode(prompt))):
            if token not in exclude_tokens:
                if token in logit_bias:
                    if use_frequency_penalty:
                        logit_bias[token] += bias * decay ** i
                else:
                    logit_bias[token] = bias * decay ** i
        for i, token in enumerate(reversed(self.tokenizer.encode(prompt.upper()))):
            if token not in exclude_tokens:
                if token in logit_bias:
                    pass # don't re-penalize tokens based on this upper/lowercase heuristic
                else:
                    logit_bias[token] = bias
        for i, token in enumerate(reversed(self.tokenizer.encode(prompt.lower()))):
            if token not in exclude_tokens:
                if token in logit_bias:
                    pass # don't re-penalize tokens based on this upper/lowercase heuristic
                else:
                    logit_bias[token] = bias
        return logit_bias

    def fit(self, dataset):
        pass
    
    def save(self, path):
        pass

    def load(self, path):
        pass

    def add_controller(self, controller):
        raise NotImplementedError