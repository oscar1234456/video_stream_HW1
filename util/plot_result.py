from sklearn.metrics import roc_auc_score, roc_curve


def Count_ROC_Curve(y_test, pos_probs):
    score = roc_auc_score(y_test, pos_probs)
    fpr_test, tpr_test, _ = roc_curve(y_test, pos_probs)

    return score, fpr_test, tpr_test