from abc import ABC, abstractmethod

class Dataset(ABC):
    @abstractmethod
    def __init__(self, args):
        pass

    @abstractmethod
    def shuffle(self, split, seed=None):
        pass

    @abstractmethod
    def load_long_texts(self, split='train', limit=None, split_paragraphs=False):
        pass

    @abstractmethod
    def load_short_texts(self, split='train', limit=None, split_paragraphs=False):
        pass

    @abstractmethod
    def pandas_format(self, split, long_name='content', short_name='title', limit=None):
        pass