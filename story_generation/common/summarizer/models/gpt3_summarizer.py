from re import T
import time
import logging

from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer
import openai

from story_generation.common.summarizer.models.abstract_summarizer import AbstractSummarizer
from story_generation.common.data.split_paragraphs import split_paragraphs, cut_last_sentence

GPT3_SEP = '\n\n###\n\n'
GPT3_END = 'THE END.'
PRETRAINED_MODELS = ['ada', 'babbage', 'curie', 'davinci', 'text-ada-001', 'text-babbage-001', 'text-curie-001', 'text-davinci-001', 'text-davinci-002']

class GPT3Summarizer(AbstractSummarizer):
    def __init__(self, args):
        assert args.gpt3_model is not None
        self.model = args.gpt3_model
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.args = args
        self.controller = None

    @torch.no_grad()
    def __call__(self, texts, generation_max_length=None, top_p=None, temperature=None, coherence_prefixes=None, retry_until_success=True, verbose=False, nodes=None, stop=None, modify_prompt=False, logit_bias={}, num_completions=1, cut_sentence=False, model_string=None):
        assert type(texts) == list
        if modify_prompt:
            logging.warning('Warning: modifying prompt for summarization')
        if model_string is None:
            logging.warning('model string not provided, using default model')
        if self.controller is None:
            return self._call_helper(texts, generation_max_length=generation_max_length, top_p=top_p, temperature=temperature, retry_until_success=retry_until_success, nodes=nodes, stop=stop, modify_prompt=modify_prompt, logit_bias=logit_bias, num_completions=num_completions, cut_sentence=cut_sentence, model_string=model_string)
        else:
            assert coherence_prefixes is not None and len(coherence_prefixes) == len(texts)
            if self.args.sentence_coherence_control_mode == 'rerank':
                generations = []
                for text, prefix in zip(texts, coherence_prefixes):
                    candidates = []
                    for _ in range(self.args.summarizer_beam_size):
                        candidates += self._call_helper([text], generation_max_length=generation_max_length, top_p=top_p, temperature=temperature, retry_until_success=retry_until_success,nodes=nodes, stop=stop, modify_prompt=modify_prompt, logit_bias=logit_bias, num_completions=num_completions, cut_sentence=cut_sentence, model_string=model_string)
                    coherence_scores = self.controller.evaluate_full_texts([prefix + c for c in candidates], reduce='none', add_prefix=False)
                    generations.append(candidates[np.argmax(coherence_scores)])
                return generations
            elif self.args.sentence_coherence_control_mode == 'greedy_sentence':
                raise NotImplementedError
            elif self.args.sentence_coherence_control_mode == 'beam_sentence':
                raise NotImplementedError
            else:
                raise NotImplementedError
    
    @torch.no_grad()
    def _call_helper(self, texts, generation_max_length=None, top_p=None, temperature=None, retry_until_success=True, nodes=None, stop=None, modify_prompt=False, logit_bias={}, num_completions=1, cut_sentence=False, model_string=None):
        given_stop = stop
        if nodes is not None:
            assert self.args.expander
            assert len(nodes) == len(texts)

        outputs = []
        for i in range(len(texts)):
            text = texts[i]
            if not modify_prompt:
                prompt = text
                stop = None if self.model in PRETRAINED_MODELS else GPT3_END
            elif nodes is not None:
                node = nodes[i]
                prompt = node.recursive_context_prompt()
                stop = '"""'
            else:
                if self.model in PRETRAINED_MODELS:
                    if self.args.expander: # generate
                        prompt = 'A summary of a story:\n"""\n' + text.strip() + '\n"""\nFull version:\n"""\n'
                    else: # summarize
                        prompt = 'A passage from a story:\n"""\n' + text.strip() + '\n"""\nOne-sentence summary:\n"""\n'
                    stop = '"""'
                else:
                    prompt = text.strip() + GPT3_SEP # finetuned model
                    stop = GPT3_END
            if given_stop is not None:
                stop = given_stop
                        
            retry = True
            num_fails = 0
            while retry:
                try:
                    context_length = len(self.tokenizer.encode(prompt))
                    if context_length > self.args.max_context_length:
                        logging.warning('context length' + ' ' + context_length + ' ' + 'exceeded artificial context length limit' + ' ' + self.args.max_context_length)
                        time.sleep(5) # similar interface to gpt3 query failing and retrying
                        assert False
                    if generation_max_length is None:
                        generation_max_length = min(self.args.generation_max_length, self.args.max_context_length - context_length)
                    engine = self.model if model_string is None else model_string
                    if engine == 'text-davinci-001':
                        engine = 'text-davinci-002' # update to latest version
                    if engine in PRETRAINED_MODELS:
                        logging.log(21, 'PROMPT')
                        logging.log(21, prompt)
                        logging.log(21, 'MODEL STRING:' + ' ' + self.model if model_string is None else model_string)
                        completion = openai.Completion.create(
                            engine=engine,
                            prompt=prompt,
                            max_tokens=generation_max_length,
                            temperature=temperature if temperature is not None else self.args.summarizer_temperature,
                            top_p=top_p if top_p is not None else self.args.summarizer_top_p,
                            frequency_penalty=self.args.summarizer_frequency_penalty,
                            presence_penalty=self.args.summarizer_presence_penalty,
                            stop=stop,
                            logit_bias=logit_bias,
                            n=num_completions)
                    else:
                        logging.log(21, 'PROMPT')
                        logging.log(21, prompt)
                        logging.log(21, 'MODEL STRING:' + ' ' + self.model if model_string is None else model_string)
                        completion = openai.Completion.create(
                            model=engine,
                            prompt=prompt,
                            max_tokens=generation_max_length,
                            temperature=temperature if temperature is not None else self.args.summarizer_temperature,
                            top_p=self.args.summarizer_top_p,
                            frequency_penalty=self.args.summarizer_frequency_penalty,
                            presence_penalty=self.args.summarizer_presence_penalty,
                            stop=stop,
                            logit_bias=logit_bias,
                            n=num_completions)
                    retry = False
                except Exception as e: 
                    logging.warning(str(e))
                    retry = retry_until_success
                    num_fails += 1
                    if num_fails > 20:
                        raise e
                    if retry:
                        logging.warning('retrying...')
                        time.sleep(num_fails)
            outputs += [completion['choices'][j]['text'] for j in range(num_completions)]
        if cut_sentence:
            for i in range(len(outputs)):
                if len(outputs[i].strip()) > 0:
                    outputs[i] = cut_last_sentence(outputs[i])
        engine = self.model if model_string is None else model_string
        logging.log(21, 'OUTPUTS')
        logging.log(21, str(outputs))
        logging.log(21, 'GPT3 CALL' + ' ' + engine + ' ' + str(len(self.tokenizer.encode(texts[0])) + sum([len(self.tokenizer.encode(o)) for o in outputs])))
        return outputs

    @torch.no_grad()
    def next_sentence(self, text, stop=None):
        stop = ['.', '!', '?'] if stop is None else ['.', '!', '?'] + [stop]
        return self(text, stop=stop, modify_prompt=False)
    
    def generate_with_prompt_repetition_penalty(self, prompt, penalty=0, stop=None, bias_component=''):
        logit_bias = {key: -penalty for key in set(self.tokenizer.encode(bias_component))} if penalty != 0 else {}
        return self._call_helper([prompt], stop=stop, modify_prompt=False, logit_bias=logit_bias)

    def fit(self, dataset):
        pass
    
    def save(self, path):
        pass

    def load(self, path):
        pass

    def add_controller(self, controller):
        assert len(controller) == 1
        self.controller = controller[0]
        assert self.controller.type == 'sentence'