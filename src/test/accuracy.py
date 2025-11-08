
def show_errors():


def mean_absolute_error(predicted, source, count):
    error_sum = 0
    for n in range(32):
        error_sum += abs(predicted[n] - source[n])

    result = error_sum / count
    return result

def mean_square_error(predicted, source, count):
    error_sum = 0
    for n in range(32):
        error_sum += (predicted[n] - source[n]) ** 2

    result = error_sum / count
    return result

def root_mean_square_error(predicted, source, count):
    error_sum = 0
    for n in range(32):
        error_sum += (predicted[n] - source[n]) ** 2

    result = (error_sum / count) ** 0.5
    return result