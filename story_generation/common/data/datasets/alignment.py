from calendar import c
import random
import os
import csv
import pickle
import math
import string
from collections import defaultdict, namedtuple
import multiprocessing as mp

import numpy as np
from tqdm import tqdm, trange
import torch
import pandas as pd

from story_generation.common.data.datasets.abstract_dataset import Dataset
from story_generation.common.data.split_paragraphs import split_texts


class AlignmentDataset(Dataset):
    def __init__(self, args):
        print('loading data')
        random.seed(args.seed)
        self.args = args
        self.debug = args.debug
        self.batch_size = args.batch_size
        self.data_dir = args.data_dir

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.splits = {}
        df = pd.read_csv(args.data_dir, delimiter=',', quotechar='"', skipinitialspace=True)
        text1 = [text.strip().replace('\n\n\n\nSummarize this passage.\n\n\n\n', '') for text in getattr(df, 'text1').tolist()][:args.limit]
        text2 = [text.strip() for text in getattr(df, 'text2').tolist()][:args.limit]
        # each item in text1 and text2 is actually a tab-separated list, different from other datasets
        # assume longer texts come first

        assert sum(args.split_sizes) == 1
        train_end = int(len(text1) * args.split_sizes[0])
        valid_end = int(len(text1) * (args.split_sizes[0] + args.split_sizes[1]))
        self.splits['train'] = (text1[:train_end], text2[:train_end])
        self.splits['valid'] = (text1[train_end:valid_end], text2[train_end:valid_end])
        self.splits['test'] = (text1[valid_end:], text2[valid_end:])

        print('done loading data')
        print('split sizes:')
        for key in ['train', 'valid', 'test']:
            print(key, len(self.splits[key]))

    def load_long_texts(self, split='train', limit=None, split_paragraphs=False):
        texts = self.splits[split][0]
        return split_texts(texts if limit is None else texts[:limit], mode=self.args.split_long_paragraph_mode if split_paragraphs else 'none')
        
    def load_short_texts(self, split='train', limit=None, split_paragraphs=False):
        texts = self.splits[split][1]
        return split_texts(texts if limit is None else texts[:limit], mode=self.args.split_short_paragraph_mode if split_paragraphs else 'none')
        
    def pandas_format(self, split, long_name='content', short_name='title', limit=None):
        raise NotImplementedError

    def shuffle(self, split, seed=None):
        assert split in ['train', 'valid', 'test']
        if seed is not None:
            random.seed(seed)
        indices = list(range(len(self.splits[split][0])))
        random.shuffle(indices)
        self.splits[split] = ([self.splits[split][0][i] for i in indices], [self.splits[split][1][i] for i in indices])

