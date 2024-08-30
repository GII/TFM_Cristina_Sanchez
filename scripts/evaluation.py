def calculate_precision(tp: int, fp: int):
    return tp/(tp+fp)

def calculate_recall(tp: int, fp: int, fn: int):
    return tp/(tp+fp+fn)

def calculate_f1_score(precision: float, recall: float):
    return 2*precision*recall/(precision+recall)
