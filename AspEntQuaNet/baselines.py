from measures import kullback_leibler_divergence, relative_absolute_error, absolute_error
import numpy as np


def get_baseline_metrics(y_true, x):
    baselines = get_baselines(x)
    kld_cc = kullback_leibler_divergence(y_true, baselines['cc'])
    rae_cc = relative_absolute_error(y_true, baselines['cc'])
    ae_cc = absolute_error(y_true, baselines['cc'])
    cc = [kld_cc, rae_cc, ae_cc]

    kld_pcc = kullback_leibler_divergence(y_true, baselines['pcc'])
    rae_pcc = relative_absolute_error(y_true, baselines['pcc'])
    ae_pcc = absolute_error(y_true, baselines['pcc'])
    pcc = [kld_pcc, rae_pcc, ae_pcc]

    kld_acc = kullback_leibler_divergence(y_true, baselines['acc'])
    rae_acc = relative_absolute_error(y_true, baselines['acc'])
    ae_acc = absolute_error(y_true, baselines['acc'])
    acc = [kld_acc, rae_acc, ae_acc]

    kld_pacc = kullback_leibler_divergence(y_true, baselines['pacc'])
    rae_pacc = relative_absolute_error(y_true, baselines['pacc'])
    ae_pacc = absolute_error(y_true, baselines['pacc'])
    pacc = [kld_pacc, rae_pacc, ae_pacc]

    return {'cc': cc, 'pcc': pcc, 'acc': acc, 'pacc': pacc}


def get_baselines(x: dict):
    cc = get_y_pred_CC(x)
    pcc = get_y_pred_PCC(x)
    acc = get_y_pred_ACC(x)
    pacc = get_y_pred_PACC(x)

    return {'cc': cc, 'pcc': pcc, 'acc': acc, 'pacc': pacc}


def get_y_pred_CC(x: dict):
    cc = [[subdataset_stats[9], subdataset_stats[10], subdataset_stats[11]] for subdataset_stats in x['statistics']]
    return np.array(cc)


def get_y_pred_PCC(x: dict):
    pcc = [[subdataset_stats[12], subdataset_stats[13], subdataset_stats[14]] for subdataset_stats in x['statistics']]
    return np.array(pcc)


def get_y_pred_ACC(x: dict):
    acc = [[subdataset_stats[15], subdataset_stats[16], subdataset_stats[17]] for subdataset_stats in x['statistics']]
    return np.array(acc)


def get_y_pred_PACC(x: dict):
    pacc = [[subdataset_stats[18], subdataset_stats[19], subdataset_stats[20]] for subdataset_stats in x['statistics']]
    return np.array(pacc)


