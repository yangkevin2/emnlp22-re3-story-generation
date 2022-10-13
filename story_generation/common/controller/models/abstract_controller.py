from abc import ABC, abstractmethod

class AbstractController(ABC):
    @abstractmethod
    def __call__(self, lm_logits, full_decoder_input_ids, keyword_ids):
        pass

    @abstractmethod
    def reset_cache(self):
        pass

    @abstractmethod
    def fit(self, dataset):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass