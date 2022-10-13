from json import load
import math

import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk import tokenize
from transformers import AutoTokenizer

SPLIT_PARAGRAPH_MODES = ['none', 'newline', 'newline-filter', 'sentence']
split_paragraph_tokenizer = None

def load_split_paragraph_tokenizer():
    global split_paragraph_tokenizer
    if split_paragraph_tokenizer is None:
        split_paragraph_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    return split_paragraph_tokenizer


def cut_last_sentence(text): # remove possibly incomplete last sentence
    text = text.rstrip() + ' and' # possibly start a new sentence so we can delete it, if the last sentence is already complete and ended with a period
    last_sentence = split_paragraphs(text, mode='sentence')[-1].strip() # possibly incomplete, so strip it
    text = text.rstrip()[:len(text.rstrip()) - len(last_sentence)].rstrip()
    return text


def cut_first_sentence(text): # remove possibly incomplete first sentence
    first_sentence = split_paragraphs(text, mode='sentence')[0].strip() # possibly incomplete, so strip it
    text = text.lstrip()[len(first_sentence):].lstrip()
    return text


def split_paragraphs(text, mode='none'):
    """
    Split a text into paragraphs.
    """
    if mode == 'none':
        return [text.strip()]
    elif mode == 'newline':
        while '\n\n' in text:
            text = text.replace('\n\n', '\n')
        return [s.strip() for s in text.split('\n')]
    elif mode == 'newline-filter':
        while '\n\n' in text:
            text = text.replace('\n\n', '\n')
        paragraphs = text.split('\n')
        return [p.strip() for p in paragraphs if len(p.split()) > 100]
    elif mode == 'sentence':
        while '\n\n' in text:
            text = text.replace('\n\n', '\n')
        return sum([[s.strip() for s in tokenize.sent_tokenize(t)] for t in text.split('\n')], [])
    else:
        raise NotImplementedError


def group_chunks(sentences, max_chunk_length=200, sep=' ', strip=True):
    tokenizer = load_split_paragraph_tokenizer()
    tokenized_lengths = [len(s) for s in tokenizer.batch_encode_plus(sentences)['input_ids']]
    num_chunks = math.ceil(sum(tokenized_lengths) / max_chunk_length)
    length_partition = partition_list(tokenized_lengths, num_chunks)
    chunks = []
    sentence_idx = 0
    for group in length_partition:
        chunk = []
        for _ in range(len(group)):
            chunk.append(sentences[sentence_idx])
            sentence_idx += 1
        chunks.append(sep.join(chunk))
    assert sentence_idx == len(sentences)
    return [c.strip() for c in chunks]


# following function is copied from https://stackoverflow.com/questions/35517051/split-a-list-of-numbers-into-n-chunks-such-that-the-chunks-have-close-to-equal
#partition list a into k partitions
def partition_list(a, k):
    #check degenerate conditions
    if k <= 1: return [a]
    if k >= len(a): return [[x] for x in a]
    #create a list of indexes to partition between, using the index on the
    #left of the partition to indicate where to partition
    #to start, roughly partition the array into equal groups of len(a)/k (note
    #that the last group may be a different size) 
    partition_between = []
    for i in range(k-1):
        partition_between.append((i+1)*len(a)//k)
    #the ideal size for all partitions is the total height of the list divided
    #by the number of paritions
    average_height = float(sum(a))/k
    best_score = None
    best_partitions = None
    count = 0
    no_improvements_count = 0
    #loop over possible partitionings
    while True:
        #partition the list
        partitions = []
        index = 0
        for div in partition_between:
            #create partitions based on partition_between
            partitions.append(a[index:div])
            index = div
        #append the last partition, which runs from the last partition divider
        #to the end of the list
        partitions.append(a[index:])
        #evaluate the partitioning
        worst_height_diff = 0
        worst_partition_index = -1
        for p in partitions:
            #compare the partition height to the ideal partition height
            height_diff = average_height - sum(p)
            #if it's the worst partition we've seen, update the variables that
            #track that
            if abs(height_diff) > abs(worst_height_diff):
                worst_height_diff = height_diff
                worst_partition_index = partitions.index(p)
        #if the worst partition from this run is still better than anything
        #we saw in previous iterations, update our best-ever variables
        if best_score is None or abs(worst_height_diff) < best_score:
            best_score = abs(worst_height_diff)
            best_partitions = partitions
            no_improvements_count = 0
        else:
            no_improvements_count += 1
        #decide if we're done: if all our partition heights are ideal, or if
        #we haven't seen improvement in >5 iterations, or we've tried 100
        #different partitionings
        #the criteria to exit are important for getting a good result with
        #complex data, and changing them is a good way to experiment with getting
        #improved results
        if worst_height_diff == 0 or no_improvements_count > 5 or count > 100:
            return best_partitions
        count += 1
        #adjust the partitioning of the worst partition to move it closer to the
        #ideal size. the overall goal is to take the worst partition and adjust
        #its size to try and make its height closer to the ideal. generally, if
        #the worst partition is too big, we want to shrink the worst partition
        #by moving one of its ends into the smaller of the two neighboring
        #partitions. if the worst partition is too small, we want to grow the
        #partition by expanding the partition towards the larger of the two
        #neighboring partitions
        if worst_partition_index == 0:   #the worst partition is the first one
            if worst_height_diff < 0: partition_between[0] -= 1   #partition too big, so make it smaller
            else: partition_between[0] += 1   #partition too small, so make it bigger
        elif worst_partition_index == len(partitions)-1: #the worst partition is the last one
            if worst_height_diff < 0: partition_between[-1] += 1   #partition too small, so make it bigger
            else: partition_between[-1] -= 1   #partition too big, so make it smaller
        else:   #the worst partition is in the middle somewhere
            left_bound = worst_partition_index - 1   #the divider before the partition
            right_bound = worst_partition_index   #the divider after the partition
            if worst_height_diff < 0:   #partition too big, so make it smaller
                if sum(partitions[worst_partition_index-1]) > sum(partitions[worst_partition_index+1]):   #the partition on the left is bigger than the one on the right, so make the one on the right bigger
                    partition_between[right_bound] -= 1
                else:   #the partition on the left is smaller than the one on the right, so make the one on the left bigger
                    partition_between[left_bound] += 1
            else:   #partition too small, make it bigger
                if sum(partitions[worst_partition_index-1]) > sum(partitions[worst_partition_index+1]): #the partition on the left is bigger than the one on the right, so make the one on the left smaller
                    partition_between[left_bound] -= 1
                else:   #the partition on the left is smaller than the one on the right, so make the one on the right smaller
                    partition_between[right_bound] += 1


def split_texts(texts, mode='none'):
    """
    Split a list of texts into paragraphs.
    """
    if mode == 'none':
        return texts
    return sum([split_paragraphs(text, mode=mode) for text in texts], [])
