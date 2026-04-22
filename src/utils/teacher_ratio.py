from matplotlib import pyplot as plt


def count_teacher_ratio(epoch, first_epoch, last_epoch):
    """
    :param epoch: actual epoch number
    :param first_epoch: first epoch to use mix of teacher forcing and free-running
    :param last_epoch: last epoch to use mix of teacher forcing and free-running (following epochs will have `teacher_ratio=0`)
    """

    teacher_ratio = min(1.0, max(0.0, (epoch - last_epoch) / (first_epoch - last_epoch)))

    return teacher_ratio


def show_teacher_ratio(epochs_number, first_epoch, last_epoch):
    """
    Showing teacher ratio on plot
    :param epochs_number: number of epochs to show
    :param first_epoch: first epoch to use mix of teacher forcing and free-running
    :param last_epoch: last epoch to use mix of teacher forcing and free-running (following epochs will have `teacher_ratio=0`)
    :return:
    """
    epochs = [i for i in range(0, epochs_number)]
    teacher_ratio_list = [(e, count_teacher_ratio(e, first_epoch, last_epoch)) for e in epochs]

    x = [val[0] for val in teacher_ratio_list]
    y = [val[1] for val in teacher_ratio_list]
    plt.plot(x, y)
    plt.show()
