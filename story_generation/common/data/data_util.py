from story_generation.common.data.datasets.writing_prompts import WritingPromptsDataset
from story_generation.common.data.datasets.alignment import AlignmentDataset
from story_generation.common.data.split_paragraphs import SPLIT_PARAGRAPH_MODES

DATASET_CHOICES=['writing_prompts', 'alignment']
# if providing a csv, shold give the full path to csv in data-dir. only for inference. 

def add_data_args(parser):
    parser.add_argument('--dataset', type=str, default='writing_prompts', choices=DATASET_CHOICES, help='dataset format')
    parser.add_argument('--data-dir', type=str, default='/home/yangk/data/story/writing_prompts', help='data directory')
    parser.add_argument('--split-sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1], help='train/val/test proportions for datasets where not provided')
    parser.add_argument('--summarizer-prediction-split', type=str, default='valid', help='split to use for summarizer predictions')
    parser.add_argument('--limit', type=int, default=None, help='limit the number of examples')
    parser.add_argument('--length-limit', type=int, default=1000000, help='limit the number of words per example')
    parser.add_argument('--lower-length-limit', type=int, default=0, help='limit the number of words per example')
    parser.add_argument('--summary-length-limit', type=int, default=1000000, help='limit the number of words in the summary')
    parser.add_argument('--single-sentence-summary', action='store_true', help='use single sentence summary data only')
    parser.add_argument('--split-long-paragraph-mode', type=str, default='none', choices=SPLIT_PARAGRAPH_MODES, help='split long paragraph mode')
    parser.add_argument('--split-short-paragraph-mode', type=str, default='none', choices=SPLIT_PARAGRAPH_MODES, help='split short paragraph mode')
    parser.add_argument('--extra-keywords', type=int, default=0, help='max number of extra keywords from long content to add to short content')
    parser.add_argument('--hallucinate-keywords', action='store_true', default=False, help='hallucinate keywords from short content')
    parser.add_argument('--keyword-file', type=str, default='/home/yangk/data/glove/glove.840B.300d.vocab', help='file to load keywords from')
    parser.add_argument('--keyword-temperature', type=float, default=1.0, help='temperature for keyword sampling')
    parser.add_argument('--csv-column', type=str, help='column name to use as input for csv')
    parser.add_argument('--num-workers', type=int, default=20, help='number of workers for data loading')
    return parser

def load_dataset(args):
    if args.dataset == 'writing_prompts':
        dataset = WritingPromptsDataset(args)
    elif args.dataset == 'alignment':
        dataset = AlignmentDataset(args)
    else:
        raise NotImplementedError
    return dataset