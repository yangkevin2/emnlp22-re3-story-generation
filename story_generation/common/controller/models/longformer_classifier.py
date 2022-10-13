import os

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer

from story_generation.common.controller.models.abstract_controller import AbstractController
from story_generation.common.util import AverageMeter, pad_to_max_length, pad_mask
from story_generation.common.controller.loader_util import get_loader
from story_generation.common.data.split_paragraphs import split_paragraphs
from story_generation.common.data.tree_util import START_OF_STORY, MIDDLE_OF_STORY


class LongformerClassifier(AbstractController):
    def __init__(self, args, index):
        self.type = 'sentence'
        self.index = index
        self.model_string = args.controller_model_string[index] if args.controller_model_string[index] != 'none' else 'allenai/longformer-base-4096'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args = args
        self.trained = False
        self.loader_type = self.args.loader[self.index]
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_string, num_labels=2).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_string)
        self.optimizer = AdamW(self.model.parameters(), lr=args.controller_lr)
    
    def reset_cache(self):
        pass

    @torch.no_grad()
    def evaluate_full_texts(self, texts, reduce='mean', add_prefix=True):
        # evaluate by prefix one sentence at a time
        all_scores = []
        for text in texts:
            while '\n\n' in text:
                text = text.replace('\n\n', '\n')
            text = text.replace('\n', ' ').strip() # since that's how it's like when trained
            sentences = split_paragraphs(text, mode='sentence')
            current_text = []
            if add_prefix and self.args.use_beginning_middle_tokens:
                current_text.append(START_OF_STORY)
            eval_texts, eval_sentences = [], []
            for sentence in sentences:
                while len(self.tokenizer.encode(' '.join(current_text + [sentence]))) > self.tokenizer.model_max_length:
                    if self.args.use_beginning_middle_tokens:
                        current_text = [MIDDLE_OF_STORY] + current_text[2:] # delete the special start token, then one extra sentence
                        if len(current_text) == 1:
                            break
                    else:
                        current_text = current_text[1:]
                        if len(current_text) == 0:
                            break
                eval_texts.append(' '.join(current_text))
                if len(self.tokenizer.encode(' '.join(current_text + [sentence]))) > self.tokenizer.model_max_length: # rare edge case of one super long sentence
                    eval_sentences.append(self.tokenizer.decode(self.tokenizer.encode(' '.join(current_text + [sentence]))[:self.tokenizer.model_max_length]))
                else:
                    eval_sentences.append(sentence.strip())
                current_text.append(sentence.strip())
            scores = self(eval_texts, eval_sentences) # should get scores or logprobs
            all_scores.append(scores.mean().item())
        if reduce == 'mean':
            return np.mean(all_scores)
        elif reduce == 'none':
            return all_scores
        else:
            raise NotImplementedError

    @torch.no_grad()
    def __call__(self, texts, sentences):
        assert len(texts) == len(sentences)
        all_texts = []
        for text, sentence in zip(texts, sentences):
            while '\n\n' in text:
                text = text.replace('\n\n', '\n')
            text = text.replace('\n', ' ').strip() # since that's how it's like when trained
            text = text + ' ' + sentence
            all_texts.append(text.strip())
        batch = self.tokenizer(all_texts, return_tensors="pt", padding=True)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        logits = outputs.logits
        positive_log_probs = F.softmax(logits, dim=-1)[:, 1].log()
        return positive_log_probs
    
    @torch.no_grad()
    def evaluate_overall_texts(self, texts):
        batch = self.tokenizer(texts, return_tensors="pt", padding=True)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        logits = outputs.logits
        positive_log_probs = F.softmax(logits, dim=-1)[:, 1].log()
        return positive_log_probs
    
    def fit(self, dataset):
        best_val_loss = 1e8
        for epoch in range(self.args.controller_epochs):
            dataset.shuffle('train')
            train_loader = get_loader(self.args.loader[self.index], 
                                      dataset, 
                                      'train', 
                                      longformer_classifier_collate, 
                                      batch_size=self.args.batch_size, 
                                      append_mask_token=False, 
                                      tokenizer_model=self.model_string, 
                                      num_workers=self.args.num_workers, 
                                      time_label_decay=self.args.fudge_time_label_decay,
                                      generate_negatives=True,
                                      num_negatives=self.args.controller_num_negatives,
                                      negative_categories=self.args.coherence_negative_categories,
                                      use_special_tokens=self.args.use_beginning_middle_tokens)
            loop = tqdm(train_loader, leave=True)
            loss_meter = AverageMeter('loss', ':6.4f')
            for batch in loop:
                # initialize calculated gradients (from prev step)
                self.optimizer.zero_grad()
                # pull all tensor batches required for training
                input_ids = batch['input_ids'].to(self.device)
                if input_ids.shape[0] < self.args.batch_size: # don't do the last batch if smaller
                    continue
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                # process
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                # update parameters
                self.optimizer.step()
                loss_meter.update(loss.detach().item(), input_ids.shape[0])
                # print relevant info to progress bar
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())
            print('Training epoch {} average loss {}'.format(epoch, loss_meter.avg))
            
            valid_loader = get_loader(self.args.loader[self.index], 
                                      dataset, 
                                      'valid', 
                                      longformer_classifier_collate, 
                                      batch_size=self.args.batch_size, 
                                      append_mask_token=False, 
                                      tokenizer_model=self.model_string, 
                                      num_workers=self.args.num_workers, 
                                      time_label_decay=self.args.fudge_time_label_decay,
                                      generate_negatives=True,
                                      num_negatives=self.args.controller_num_negatives,
                                      negative_categories=self.args.coherence_negative_categories,
                                      use_special_tokens=self.args.use_beginning_middle_tokens)
            loop = tqdm(valid_loader, leave=True)
            loss_meter = AverageMeter('loss', ':6.4f')
            with torch.no_grad():
                for batch in loop:
                    # pull all tensor batches required for training
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    # process
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss_meter.update(loss.item(), input_ids.shape[0])
                    # print relevant info to progress bar
                    loop.set_description(f'Epoch {epoch}')
                    loop.set_postfix(loss=loss.item())
                print('Validation epoch {} average loss {}'.format(epoch, loss_meter.avg))
            if loss_meter.avg < best_val_loss:
                print('Found new best model. Saving...')
                best_val_loss = loss_meter.avg
                self.save(os.path.join(self.args.controller_save_dir, 'model_best.pth.tar'))

        self.trained = True

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'args': self.args
        }, path)

    def load(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
        except:
            checkpoint = torch.load(os.path.join(path, 'model_best.pth.tar'), map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.trained = True


def longformer_classifier_collate(batch):
    batch = sum(batch, [])
    lengths = torch.LongTensor([len(p['prefix']) for p in batch])
    inputs = [torch.LongTensor(p['prefix']) for p in batch]
    input_ids = torch.stack(pad_to_max_length(inputs, 0), dim=0)
    attention_mask = pad_mask(lengths).permute(1, 0)
    labels = torch.stack([torch.from_numpy(p['labels']) for p in batch], dim=0)
    return {'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'labels': labels, 
            'lengths': lengths}