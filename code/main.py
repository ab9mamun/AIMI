# -*- coding: utf-8 -*-

import random
import os
from experimenter.experimenter import run
import argparse
from sklearn.utils import shuffle
def set_seeds(seed):
    """
    Set random seeds for reproducibility.
    :param seed:
    :return:
    """
    import tensorflow as tf
    print(tf.__version__)
    tf.keras.utils.set_random_seed(seed)

def run_experiment(basepath, dataset, data_path, output_path, arch, num_epochs, user_choices, trial, seed):

    left_pivots = [0, 1, 5, 9, 13, 17]
    right_pivots = [1, 5, 9, 13, 17, 22]

    for subfeatures in ['high_res', 'high_and_knowledge', 'without_knowledge', 'all', 'location_only', 'location_and_knowledge']:
        pretrained_file = None
        for i in range(len(left_pivots)):
            left_pivot = left_pivots[i]
            right_pivot = right_pivots[i]
            users = user_choices[left_pivot:right_pivot]
            usertext = '_'.join([str(user) for user in users])
            print('Trial:', trial, 'Seed:', seed, 'Subfeatures:', subfeatures, 'Users:', usertext)
            if os.path.exists(f'{output_path}/trial_{trial}_pivots_{left_pivot}_{right_pivot}_predictions_{subfeatures}_{usertext}_SEED_{seed}.csv'):
                print('Results are already there.')
            else:
                pretrained_file = run(basepath,dataset, data_path, output_path, trial, seed, arch, num_epochs, users, subfeatures, pretrained_file, left_pivot, right_pivot)
def main(args):
    """
    Main function to run the experiment.
    :return:
    """

    args = parser.parse_args()
    seed = args.seed
    set_seeds(seed)
    arch = args.arch
    num_epochs = args.num_epochs
    exp_name = args.exp_name
    task = args.task

    print('Experiment name: ', exp_name)

    basepath = '/base_parent_path/hfailure_project' # replace base_parent_path with the path to the folder for data.
    data_path = f'/base_parent_path/hfailure_project/data/{args.dataset}'
    output_path = f'/base_parent_path/hfailure_project/output/{exp_name}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print('Output path: ', output_path)


    if task == 'train_all':
        all_possible_users  =[116, 117, 118, 120, 121, 122, 123, 124, 125, 126,
                              127, 129, 131, 132, 133,134, 135, 136, 140, 141, 142,
                              143]

        user_order1 = shuffle(all_possible_users, random_state=seed+1)
        user_order2 = shuffle(all_possible_users, random_state=seed+2)
        user_order3 = shuffle(all_possible_users, random_state=seed+3)

        run_experiment(basepath, args.dataset, data_path, output_path, arch, num_epochs, user_order1, 1, seed)
        run_experiment(basepath, args.dataset, data_path, output_path, arch, num_epochs, user_order2, 2, seed)
        run_experiment(basepath, args.dataset, data_path, output_path, arch, num_epochs, user_order3, 3, seed)

    else:
        print("Unknown task")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adherence Forecasting Project.')

    parser.add_argument('--dataset',
                        default='nih',
                        type=str)
    parser.add_argument('--exp_name',
                        default='default_exp_name',
                        help='A unique name for the experiment. If not unique, the existing experiment may be overwritten.',
                        type=str)
    parser.add_argument('--task',
                        default='None',
                        help='Choose from train_all',
                        type=str)
    parser.add_argument('--seed',
                        default=42,
                        help='Seed for random number generation',
                        type=int)

    parser.add_argument('--num_epochs',
                        default=2,
                        help='Number of epochs for training',
                        type=int)

    parser.add_argument('--arch',
                        default='None',
                        help='Type of model to use. Choose from cnn, lstm',
                        type=str)

    main(args=parser.parse_args())
    
