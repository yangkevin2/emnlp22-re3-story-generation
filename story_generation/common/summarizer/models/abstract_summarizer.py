from abc import ABC, abstractmethod

class AbstractSummarizer(ABC):
    @abstractmethod
    def __call__(self, texts):
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

    @abstractmethod
    def add_controller(self, controller):
        pass