

def compute_metrics(p):
    predictions, labels = p


    results = {
        'recall': recall,
        'precision': precision,
        'f5': f1_score
    }
    return results
