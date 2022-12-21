from story_generation.common.summarizer.models.gpt3_summarizer import GPT3Summarizer
from story_generation.common.summarizer.models.opt_summarizer import OPTSummarizer

SUMMARIZER_CHOICES=['gpt3_summarizer', 'opt_summarizer']

def add_summarizer_args(parser):
    parser.add_argument('--summarizer', type=str, default='gpt3_summarizer', choices=SUMMARIZER_CHOICES, help='model architecture')
    parser.add_argument('--summarizer-save-dir', type=str, default=None, help='directory to save summarizer')
    parser.add_argument('--summarizer-load-dir', type=str, default=None, help='directory to load summarizer')
    parser.add_argument('--expander', action='store_true', help='swap source and target to learn expanding a summary')
    parser.add_argument('--summarizer-temperature', type=float, default=0.8, help='temperature for summarizer')
    parser.add_argument('--opt-summarizer-temperature', type=float, default=0.8, help='temperature for OPT summarizer during main story generation')
    parser.add_argument('--summarizer-top-p', type=float, default=1.0, help='top p for summarizer')
    parser.add_argument('--summarizer-frequency-penalty', type=float, default=0.5, help='frequency penalty for summarizer')
    parser.add_argument('--summarizer-prompt-penalty', type=float, default=0.5, help='OPT control penalty for prompt tokens for summarizer, excluding stopwords/punc/names')
    parser.add_argument('--summarizer-frequency-penalty-decay', type=float, default=0.98, help='frequency penalty decay for OPT summarizer')
    parser.add_argument('--summarizer-presence-penalty', type=float, default=0, help='presence penalty for summarizer')
    parser.add_argument('--generation-max-length', type=int, default=256, help='max length for generation, not including prompt')
    parser.add_argument('--summarizer-beam-size', type=int, default=1, help='beam size for summarizer')
    parser.add_argument('--gpt3-model', type=str, default='text-davinci-002', help='gpt3 model or finetuned ckpt for GPT3Summarizer')
    parser.add_argument('--max-context-length', type=int, default=1024, help='max length for context to facilitate toy version')
    parser.add_argument('--alpa-url', type=str, default=None, help='url for alpa API')
    parser.add_argument('--alpa-port', type=str, default=None, help='port for alpa API, if alpa-url is a filename to read server location from. convenient for slurm')
    parser.add_argument('--alpa-key', type=str, default='', help='key for alpa API, if using the public API')
    return parser

def load_summarizer(args):
    if args.summarizer == 'gpt3_summarizer':
        summarizer = GPT3Summarizer(args)
    elif args.summarizer == 'opt_summarizer':
        summarizer = OPTSummarizer(args)
    else:
        raise NotImplementedError
    return summarizer