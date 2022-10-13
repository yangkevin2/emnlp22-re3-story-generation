import torch

from story_generation.common.controller.loaders.coherence_loader import CoherenceSplitLoader
from story_generation.common.controller.loaders.alignment_loader import AlignmentSplitLoader

def get_loader(loader_name, dataset, split, collate_fn, batch_size=32, append_mask_token=False, num_workers=20, tokenizer_model='roberta-base', time_label_decay=1, **kwargs):
    assert split in ['train', 'valid', 'test']
    if loader_name == 'coherence':
        loader_class = CoherenceSplitLoader
    elif loader_name == 'alignment':
        loader_class = AlignmentSplitLoader
    else:
        raise NotImplementedError
    print('loading long short texts for data loader')
    contents, summaries = dataset.load_long_texts(split, split_paragraphs=False), dataset.load_short_texts(split, split_paragraphs=False)
    print('done loading long short texts')
    return torch.utils.data.DataLoader(loader_class(contents, summaries, tokenizer_model, append_mask_token=False, time_label_decay=time_label_decay, **kwargs), batch_size=batch_size, pin_memory=True, collate_fn=collate_fn, num_workers=num_workers)
