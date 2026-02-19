from dtaidistance import dtw
from nltk.metrics import edit_distance
from tslearn.metrics import dtw as ts_dtw
import similaritymeasures

def dynamic_time_warping_score():
    time_series_a = [1, 2, 2, 3]
    time_series_b = [1, 4, 3]

    distance, paths = dtw.warping_paths(time_series_a, time_series_b, use_c=False)
    best_path = dtw.best_path(paths)
    print(distance)
    print(best_path)

    similarity_score = distance / len(best_path)
    print(similarity_score)

    return similarity_score

def edit_distance_multi_col():
    seq_a = [(1, 1, 1), (1, 2, 2), (4, 1, 1), (2, 1, 0)]
    seq_b = [(1, 1, 1), (1, 3, 3), (4, 1, 1), (4, 1, 1), (2, 1, 1)]

    score = edit_distance(seq_a, seq_b)
    print(score)

    return score

def discrete_frechet():
    seq_a = [(1, 1, 1), (1, 2, 2), (4, 1, 1), (2, 1, 0)]
    seq_b = [(1, 1, 1), (1, 3, 3), (4, 1, 1), (4, 1, 1), (2, 1, 1)]

    score = similaritymeasures.frechet_dist(
        seq_a,
        seq_b
    )

    print(score)

    return score

if __name__ == '__main__':
    discrete_frechet()