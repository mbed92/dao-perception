from argparse import ArgumentParser

import yaml
import world

def generate(args):
    config = yaml.safe_load(open(args.config_file, 'r'))
    train_data, val_data, test_data = world.elastica.generator.randomized_dataset(config)
    print(config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config-file', type=str, default="../config/generate_dataset.yaml")
    args, _ = parser.parse_known_args()
    generate(args)
