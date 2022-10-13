import os
import csv
import json
import random

START_OF_STORY = '[Beginning of story]'
MIDDLE_OF_STORY = '[Previous details omitted] ...'
SHORT_TEXT_PROMPT = 'Elaborate on the following passage:'
LONG_TEXT_PROMPT = 'Story context:'
PRE_TEXT_SEP = '\n\n"""\n'
POST_TEXT_SEP = '\n"""\n\n'


class Node:
    def __init__(self):
        self.children = []
        self.parent = None
        self.long_text = None
        self.short_text = None
    
    def set_long_text(self, long_text):
        self.long_text = long_text
        return self
    
    def set_short_text(self, short_text):
        self.short_text = short_text
        return self
    
    def add_parent(self, parent=None):
        self.parent = Node() if parent is None else parent
        self.parent.children.append(self)
        return self.parent
    
    def add_child(self, child=None):
        child = Node() if child is None else child
        child.parent = self
        self.children.append(child)
        return child
    
    def max_depth_from_self(self):
        if len(self.children) == 0:
            return 0
        else:
            depths = [child.max_depth_from_self() for child in self.children]
            return max(depths) + 1
    
    def ordered_leaves(self):
        if len(self.children) == 0:
            return [self]
        else:
            leaves = []
            for child in self.children:
                leaves += child.ordered_leaves()
            return leaves
    
    def traverse_subtree(self):
        yield self
        for child in self.children:
            yield from child.traverse_subtree()
    
    def depth(self):
        if self.parent is None:
            return 0
        else:
            return 1 + self.parent.depth()
    
    def nodes_at_depth(self, depth):
        if depth == 0:
            return [self]
        else:
            nodes = []
            for child in self.children:
                nodes += child.nodes_at_depth(depth-1)
            return nodes
    
    def root(self):
        current_node = self
        while current_node.parent is not None:
            current_node = current_node.parent
        return current_node
    
    def previous_same_depth_node(self):
        same_depth_nodes = self.root().nodes_at_depth(self.depth())
        idx = same_depth_nodes.index(self)
        return same_depth_nodes[idx-1] if idx > 0 else None
    
    def coherence_prefix(self):
        previous_same_depth_node = self.previous_same_depth_node()
        prefix = START_OF_STORY if (previous_same_depth_node is None or previous_same_depth_node.previous_same_depth_node() is None) else MIDDLE_OF_STORY
        # separate with ' ' for short text, '\n' for long text for sentences vs paragraphs, maybe
        previous_long_text = prefix + ' ' + previous_same_depth_node.long_text.strip() if previous_same_depth_node is not None else prefix
        return previous_long_text
    
    def context_expansion(self):
        previous_same_depth_node = self.previous_same_depth_node()
        prefix = START_OF_STORY if (previous_same_depth_node is None or previous_same_depth_node.previous_same_depth_node() is None) else MIDDLE_OF_STORY
        # separate with ' ' for short text, '\n' for long text for sentences vs paragraphs, maybe
        previous_short_text = prefix + ' ' + previous_same_depth_node.short_text.strip() + ' ' if previous_same_depth_node is not None else prefix + ' '
        previous_long_text = prefix + ' ' + previous_same_depth_node.long_text.strip() + ' ' if previous_same_depth_node is not None else prefix + ' '
        expansion = SHORT_TEXT_PROMPT + PRE_TEXT_SEP + previous_short_text + self.short_text.strip() + POST_TEXT_SEP + LONG_TEXT_PROMPT + PRE_TEXT_SEP + previous_long_text + self.long_text.strip() + POST_TEXT_SEP
        return expansion

    def recursive_context_prompt(self):
        previous_same_depth_node = self.previous_same_depth_node()
        prefix = START_OF_STORY if (previous_same_depth_node is None or previous_same_depth_node.previous_same_depth_node() is None) else MIDDLE_OF_STORY
        # separate with ' ' for short text, '\n' for long text for sentences vs paragraphs, maybe
        previous_short_text = prefix + ' ' + previous_same_depth_node.short_text.strip() + ' ' if previous_same_depth_node is not None else prefix + ' '
        previous_long_text = prefix + ' ' + previous_same_depth_node.long_text.strip() if previous_same_depth_node is not None else prefix
        prompt = SHORT_TEXT_PROMPT + PRE_TEXT_SEP + previous_short_text + self.short_text.strip() + POST_TEXT_SEP + LONG_TEXT_PROMPT + PRE_TEXT_SEP + previous_long_text
        
        current_node = self
        while current_node.parent is not None:
            current_node = current_node.parent
            prompt = current_node.context_expansion() + prompt
        return prompt
    
    def context_completion(self):
        return ' ' + self.long_text.strip() + POST_TEXT_SEP
    
    def full_text(self, joiner=' '):
        return joiner.join([leaf.long_text for leaf in self.ordered_leaves()])


def save_gpt3_prompts(trees, path, shuffle=True):
    all_data_dicts = []
    for root in trees:
        for node in root.traverse_subtree():
            prompt = node.recursive_context_prompt()
            completion = node.context_completion()
            all_data_dicts.append({'prompt': prompt, 'completion': completion})
    if shuffle:
        random.shuffle(all_data_dicts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as wf:
        for data_dict in all_data_dicts:
            wf.write(json.dumps(data_dict) + '\n')


def save_trees(trees, path, mode='all', replace_newline=True, joiner='***', short_long_sep='@@@'):
    assert mode in ['all', 'final_long', 'final_short']
    num_iterations = max([root.max_depth_from_self() for root in trees])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as wf:
        writer = csv.writer(wf)
        if mode == 'final_short':
            writer.writerow(['final_short'])
            for root in trees:
                writer.writerow([root.short_text.replace('\n', '\\n') if replace_newline else root.short_text])
        elif mode == 'final_long':
            writer.writerow(['final_long'])
            for root in trees:
                long_text = root.full_text(joiner=joiner)
                long_text = long_text.replace('\n', '\\n') if replace_newline else long_text
                writer.writerow([long_text])
        elif mode == 'all':
            writer.writerow(['iter' + str(i) for i in range(num_iterations+1)])
            for root in trees:
                iters = []
                current_nodes = [root]
                while len(current_nodes) > 0:
                    iters.append(joiner.join([node.short_text for node in current_nodes]) + short_long_sep + joiner.join([node.long_text for node in current_nodes]))
                    current_nodes = sum([node.children for node in current_nodes], [])
                writer.writerow([t.replace('\n', '\\n') for t in iters] if replace_newline else iters)