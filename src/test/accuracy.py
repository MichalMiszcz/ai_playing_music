
def show_errors(predicted, source, count):
    errors_index = []
    for n in range(count):
        if predicted[n] != source[n]:
            errors_index.append(n)

    return errors_index

def percent_correct(predicted, source, count):
    errors_count = 0
    for n in range(count):
        if predicted[n] != source[n]:
            errors_count += 1

    correct_percent = 100 - (errors_count / count) * 100

    return correct_percent

def mean_absolute_error(predicted, source, count):
    error_sum = 0
    for n in range(count):
        error_sum += abs(predicted[n] - source[n])

    result = error_sum / count
    return result

def mean_square_error(predicted, source, count):
    error_sum = 0
    for n in range(count):
        error_sum += (predicted[n] - source[n]) ** 2

    result = error_sum / count
    return result

def root_mean_square_error(predicted, source, count):
    error_sum = 0
    for n in range(count):
        error_sum += (predicted[n] - source[n]) ** 2

    result = (error_sum / count) ** 0.5
    return result