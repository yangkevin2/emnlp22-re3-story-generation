from story_generation.common.summarizer.models.gpt3_summarizer import GPT3Summarizer

SUMMARIZER_CHOICES=['gpt3_summarizer']

def add_summarizer_args(parser):
    parser.add_argument('--summarizer', type=str, default='gpt3_summarizer', choices=SUMMARIZER_CHOICES, help='model architecture')
    parser.add_argument('--summarizer-save-dir', type=str, default=None, help='directory to save summarizer')
    parser.add_argument('--summarizer-load-dir', type=str, default=None, help='directory to load summarizer')
    parser.add_argument('--expander', action='store_true', help='swap source and target to learn expanding a summary')
    parser.add_argument('--summarizer-temperature', type=float, default=0.8, help='temperature for summarizer')
    parser.add_argument('--summarizer-top-p', type=float, default=1.0, help='top p for summarizer')
    parser.add_argument('--summarizer-frequency-penalty', type=float, default=0.5, help='frequency penalty for summarizer')
    parser.add_argument('--summarizer-presence-penalty', type=float, default=0, help='presence penalty for summarizer')
    parser.add_argument('--generation-max-length', type=int, default=256, help='max length for generation, not including prompt')
    parser.add_argument('--summarizer-beam-size', type=int, default=1, help='beam size for summarizer')
    parser.add_argument('--gpt3-model', type=str, default='text-davinci-002', help='gpt3 model or finetuned ckpt for GPT3Summarizer')
    parser.add_argument('--max-context-length', type=int, default=1024, help='max length for context to facilitate toy version')
    return parser

def load_summarizer(args):
    if args.summarizer == 'gpt3_summarizer':
        summarizer = GPT3Summarizer(args)
    else:
        raise NotImplementedError
    return summarizer