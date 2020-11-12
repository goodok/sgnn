import numpy as np
import warnings


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_color(text, color='green', bold=False, underline=False):
    assert color in ['green', 'blue', 'red', 'olive', 'magenta']
    if color == 'green':
        print(bcolors.OKGREEN, end='')
    elif color == 'blue':
        print(bcolors.OKBLUE, end='')
    elif color == 'red':
        print(bcolors.FAIL, end='')
    elif color == 'olive':
        print(bcolors.WARNING, end='')
    elif color == 'magenta':
        print(bcolors.HEADER, end='')

    if bold:
        print(bcolors.BOLD, end='')
    if underline:
        print(bcolors.UNDERLINE, end='')

    print(text, end='')

    print(bcolors.ENDC)


log_options = {
    'max_name_length': 14,
    'max_shape_length': 14,
}

scalar_types = (int, np.int, np.int32, np.int64, np.uint, np.uint32, np.uint64)


def log(text, array=None, indent=''):
    """Prints a text message. And, optionally, if a Numpy array is provided iterators
    prints iterators's shape, min, and max values.
    """
    if isinstance(text, dict) or isinstance(array, dict):
        log_dict(text, array)
        return
    if isinstance(array, list):
        d = dict([(str(i), v) for i, v in enumerate(array)])
        log_dict(text, d)
        return
    if array is not None:
        text = indent + text.ljust(log_options['max_name_length'])
        # if scalar
        if isinstance(array, scalar_types):
            text += f"{array}"
        else:
            try:
                s_mean = '{:10.5f}'.format(array.mean()) if array.size else ""
            except:
                try:
                    s_mean = '{:10.5f}'.format(
                        np.array(array).mean()) if array.size else ""
                except:
                    s_mean = ""
            s_min = calc_and_format_minmax(array, 'min')
            s_max = calc_and_format_minmax(array, 'max')

            dtype = str(array.dtype)
            s_shape = '{:' + str(log_options['max_shape_length']) + '}'
            s_shape = s_shape.format(str(tuple(array.shape)))
            text += ("shape: {}  dtype: {:13}  min: {},  max: {},  mean: {}".format(
                s_shape,
                dtype,
                s_min,
                s_max,
                s_mean))
    print(text)


def log_as_dict(text, array=None, indent=''):
    r = {}
    r['title'] = indent + text
    if array is not None:
        if isinstance(array, scalar_types):
            r['value'] = array
            r['dtype'] = str(type(array))
        else:
            r['dtype'] = str(array.dtype)
            try:
                s_mean = '{:10.5f}'.format(array.mean()) if array.size else ""
            except:
                try:
                    s_mean = '{:10.5f}'.format(
                        np.array(array).mean()) if array.size else ""
                except:
                    s_mean = ""
            r['mean'] = s_mean
            r['min'] = calc_and_format_minmax(array, 'min')
            r['max'] = calc_and_format_minmax(array, 'max')
            # s_shape = '{:' + str(log_options['max_shape_length']) + '}'
            r['shape'] = str(tuple(array.shape))
    return r


def log_dict(text, d=None):
    if d is None:
        d = text
    else:
        print(f'{text}:')
    indent = ' ' * 3
    for key in d:
        try:
            value = d[key]
            s_key = '{:<' + str(log_options['max_name_length']) + '}'
            s_key = s_key.format(key)

            if isinstance(value, str):
                print(f"{indent}{s_key}'{value}'")
            elif isinstance(value, list):
                print(f"{indent}{s_key}{value}")
            else:
                log(key, value, indent=indent)
        except Exception as e:
            warnings.warn(f"Can't log key='{key}': {e}")
            # warn_always(f"Can't log key='{key}': {e}")


def calc_and_format_minmax(array, fn='min'):
    s = ''
    if array.size:
        if fn == 'min':
            v = array.min()
        else:
            v = array.max()

    if is_array_integer(array):
        s = '{:10}'.format(v)
    else:
        s = '{:10.5f}'.format(v)
    return s


def is_array_integer(a):
    s_integers = ['torch.int16', 'torch.int32', 'torch.int64',
                  'int16', 'int32', 'int64',
                  ]
    return str(a.dtype) in s_integers
