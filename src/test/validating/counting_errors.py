def count_errors(source, output):
    number_of_errors = 0
    sum_difference = 0

    for i, [note, time] in enumerate(source):
        if output[i][0] != note:
            number_of_errors += 1
            sum_difference += abs(note - output[i][0])

        if output[i][1] != time:
            number_of_errors += 1
            sum_difference += abs(time - output[i][1])

    return number_of_errors, sum_difference