def dict_to_string(d, num_dict_round_floats=6, style='dictionary'):
    """given a dictionary, returns a string

    Arguments:
        d {dictionary} --
        num_dict_round_floats {int} -- num of digits to which the float numbers on the dict should be round
        style {string}
    """
    separator = ''
    descriptor = ''
    if style == 'dictionary':
        separator = ';'
        descriptor = ': '
    elif style == 'constructor':
        separator = ',\n'
        descriptor = '='
    
    s = ''
    for key, value in d.items():
        if isinstance(value, float):
            s += '{}{}{}{} '.format(key, descriptor, round(value, num_dict_round_floats), separator)
        else:
            s += '{}{}{}{} '.format(key, descriptor, value, separator)
    return s