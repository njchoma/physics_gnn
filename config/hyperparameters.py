
def hyperparameters():
    hyperparameter = {
        'possible_modes': ['test', 'train', 'plot', 'description',
                           'prepare_data', 'weight_average'],
        'possible_stats': ['loss_avg', 'accuracy_avg', 'output_avg',
                           'output_std', 'kernel_std'],
    }
    return hyperparameter
