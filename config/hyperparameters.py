
def hyperparameters():
    hyperparameter = {
        'possible_modes': ['test', 'train', 'plot', 'description',
                           'prepare_data', 'weight_average', 'setdefault'],
        'possible_stats': ['loss', 'kernel',
                           'avg0', 'avg1',
                           'std0', 'std1'],
    }
    return hyperparameter
