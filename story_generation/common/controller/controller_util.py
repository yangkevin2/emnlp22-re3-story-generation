import torch

from story_generation.common.controller.models.longformer_classifier import LongformerClassifier

CONTROLLER_CHOICES=['longformer_classifier']
LOADER_CHOICES=['coherence', 'alignment']

def add_controller_args(parser):
    parser.add_argument('--controller', type=str, nargs='*', default=['longformer_classifier'], choices=CONTROLLER_CHOICES, help='model architecture')
    parser.add_argument('--controller-model-string', type=str,  nargs='*', default=['none'], help='model string')
    parser.add_argument('--loader', type=str, nargs='*', default=['coherence'], choices=LOADER_CHOICES, help='loader for controller')
    parser.add_argument('--controller-save-dir', type=str, default=None, help='directory to save controller')
    parser.add_argument('--controller-load-dir', type=str, nargs='*', default=[''], help='directory to load controller')
    parser.add_argument('--controller-epochs', type=int, default=1, help='number of epochs for controller finetuning')
    parser.add_argument('--fudge-time-label-decay', type=float, default=1.0, help='discounting for label weights over time for controller training')
    parser.add_argument('--control-strength', type=float, nargs='*', default=None, help='strength of control for controller inference')
    parser.add_argument('--fudge-top-k', type=int, nargs='*', default=[100], help='top k for fudge inference')
    parser.add_argument('--controller-num-negatives', type=int, default=1, help='number of negative samples for controller contrastive training')
    parser.add_argument('--coherence-negative-categories', type=str, nargs='*', default=['other', 'repeat', 'shuffle'], help='types of negatives for coherence')
    parser.add_argument('--controller-margin', type=int, default=1, help='margin for controller contrastive training')
    parser.add_argument('--hierarchical-sentence-encoder', action='store_true', help='use hierarchical sentence encoder in sentence prefix completion classifier')
    parser.add_argument('--hierarchical-sentence-position-encodings', action='store_true', help='use hierarchical sentence position encodings in sentence prefix completion classifier')
    parser.add_argument('--freeze-epochs', type=int, default=0, help='number of epochs to freeze pretrained backbone')
    parser.add_argument('--controller-lr', type=float, default=5e-5, help='learning rate for controller finetuning')
    parser.add_argument('--use-beginning-middle-tokens', action='store_true', help='use special beginning/middle tokens for coherence training')
    parser.add_argument('--coherence-eval-index', type=int, default=None, help='index of controller to use for coherence eval')
    parser.add_argument('--eval-only-controllers', type=int, nargs='*', default=[], help='indices of controllers to use for eval only')
    parser.add_argument('--sentence-coherence-control-mode', type=str, default=None, choices=['rerank', 'greedy-sentence', 'beam-sentence'], help='how to use sentence-level coherence controller for inference')
    return parser

def load_controller(args, index):
    if args.controller[index] == 'longformer_classifier':
        controller = LongformerClassifier(args, index)
        if len(args.controller_load_dir[index]) > 0:
            controller.load(args.controller_load_dir[index])
    else:
        raise NotImplementedError
    return controller