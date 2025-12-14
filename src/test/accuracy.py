from dtaidistance import dtw

def dynamic_time_warping_score(df_predicted, df_source, column: str):
    time_series_a = df_predicted[[column]].to_numpy()
    time_series_b = df_source[[column]].to_numpy()

    distance, paths = dtw.warping_paths(time_series_a, time_series_b, use_c=False)
    best_path = dtw.best_path(paths)

    similarity_score = distance / len(best_path)
    return similarity_score

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