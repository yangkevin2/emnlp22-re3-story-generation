import argparse

from story_generation.common.util import add_general_args
from story_generation.common.data.data_util import add_data_args, load_dataset
from story_generation.common.controller.controller_util import add_controller_args, load_controller

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser = add_general_args(parser)
    parser = add_data_args(parser)
    parser = add_controller_args(parser)
    args = parser.parse_args()

    assert args.controller_save_dir is not None

    controller = load_controller(args, 0)
    dataset = load_dataset(args)
    dataset.shuffle('train')
    controller.fit(dataset)
