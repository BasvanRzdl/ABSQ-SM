import numpy as np


def confusion_matrix(y_true, y_pred):
    cm = np.zeros((3, 3))
    for p,t in zip(y_pred, y_true):
        if p[0] == 1:
            if t[0] == 1: cm[0][0] = cm[0][0] + 1
            if t[1] == 1: cm[0][1] = cm[0][1] + 1
            if t[2] == 1: cm[0][2] = cm[0][2] + 1
        if p[1] == 1:
            if t[0] == 1: cm[1][0] = cm[1][0] + 1
            if t[1] == 1: cm[1][1] = cm[1][1] + 1
            if t[2] == 1: cm[1][2] = cm[1][2] + 1
        if p[2] == 1:
            if t[0] == 1: cm[2][0] = cm[2][0] + 1
            if t[1] == 1: cm[2][1] = cm[2][1] + 1
            if t[2] == 1: cm[2][2] = cm[2][2] + 1
    return cm

def calculate_classifications(ty_train, py_train, prob_train):
    # Convert to the correct format
    ty_train = np.asarray(ty_train)
    py_train = np.asarray(py_train)
    prob_train = np.asarray(prob_train)
    cc_p1, cc_p2, cc_p3, cc_f12, cc_f13, cc_f21, cc_f23, cc_f31, cc_f32 = 0, 0, 0, 0, 0, 0, 0, 0, 0
    pcc_p1, pcc_p2, pcc_p3, pcc_f12, pcc_f13, pcc_f21, pcc_f23, pcc_f31, pcc_f32 = 0, 0, 0, 0, 0, 0, 0, 0, 0
    # increase the counts when needed
    for index in range(len(ty_train)):
        if ty_train[index] == 0 and py_train[index] == 0:
            cc_p1 += 1
        if ty_train[index] == 0 and py_train[index] == 1:
            cc_f21 += 1
        if ty_train[index] == 0 and py_train[index] == 2:
            cc_f31 += 1
        if ty_train[index] == 1 and py_train[index] == 0:
            cc_f12 += 1
        if ty_train[index] == 1 and py_train[index] == 1:
            cc_p2 += 1
        if ty_train[index] == 1 and py_train[index] == 2:
            cc_f32 += 1
        if ty_train[index] == 2 and py_train[index] == 0:
            cc_f13 += 1
        if ty_train[index] == 2 and py_train[index] == 1:
            cc_f23 += 1
        if ty_train[index] == 2 and py_train[index] == 2:
            cc_p3 += 1
    # also for probabilistic
    for index in range(len(ty_train)):
        if ty_train[index] == 0:
            pcc_p1 += prob_train[index][0]
            pcc_f21 += prob_train[index][1]
            pcc_f31 += prob_train[index][2]
        if ty_train[index] == 1:
            pcc_f12 += prob_train[index][0]
            pcc_p2 += prob_train[index][1]
            pcc_f32 += prob_train[index][2]
        if ty_train[index] == 2:
            pcc_f13 += prob_train[index][0]
            pcc_f23 += prob_train[index][1]
            pcc_p3 += prob_train[index][2]

    # calculate the true/false preediction rates
    CC_T1 = (cc_p1) / (cc_p1 + cc_f21 + cc_f31 + 0.0001)
    CC_T2 = (cc_p2) / (cc_p2 + cc_f12 + cc_f32 + 0.0001)
    CC_T3 = (cc_p3) / (cc_p3 + cc_f23 + cc_f23 + 0.0001)
    CC_F12 = (cc_f12) / (cc_f12 + cc_p2 + cc_f32 + 0.0001)
    CC_F13 = (cc_f13) / (cc_f13 + cc_f23 + cc_p3 + 0.0001)
    CC_F21 = (cc_f21) / (cc_p1 + cc_f21 + cc_f31 + 0.0001)
    CC_F23 = (cc_f23) / (cc_f13 + cc_f23 + cc_p3 + 0.0001)
    CC_F31 = (cc_f31) / (cc_p1 + cc_f21 + cc_f31 + 0.0001)
    CC_F32 = (cc_f32) / (cc_f12 + cc_p2 + cc_f32 + 0.0001)

    PCC_T1 = (pcc_p1) / (pcc_p1 + pcc_f21 + pcc_f31 + 0.0001)
    PCC_T2 = (pcc_p2) / (pcc_p2 + pcc_f12 + pcc_f32 + 0.0001)
    PCC_T3 = (pcc_p3) / (pcc_p3 + pcc_f23 + pcc_f23 + 0.0001)
    PCC_F12 = (pcc_f12) / (pcc_f12 + pcc_p2 + pcc_f32 + 0.0001)
    PCC_F13 = (pcc_f13) / (pcc_f13 + pcc_f23 + pcc_p3 + 0.0001)
    PCC_F21 = (pcc_f21) / (pcc_p1 + pcc_f21 + pcc_f31 + 0.0001)
    PCC_F23 = (pcc_f23) / (pcc_f13 + pcc_f23 + pcc_p3 + 0.0001)
    PCC_F31 = (pcc_f31) / (pcc_p1 + pcc_f21 + pcc_f31 + 0.0001)
    PCC_F32 = (pcc_f32) / (pcc_f12 + pcc_p2 + pcc_f32 + 0.0001)

    # GENERAL
    # CC_T1 = 0.815950753384304
    # CC_T2 = 0.9378316195730387
    # CC_T3 = 0.1999997777780247
    # CC_F12 = 0.056861254218252145
    # CC_F13 = 0.24999965277826003
    # CC_F21 = 0.17177910597564297
    # CC_F23 = 0.49999930555652006
    # CC_F31 = 0.012269936141117354
    # CC_F32 = 0.005307050393703534
    #
    # PCC_T1 = 0.7556937055407879
    # PCC_T2 = 0.8932328235714556
    # PCC_T3 = 0.21932709209126056
    # PCC_F12 = 0.0814296399178251
    # PCC_F13 = 0.2973829439026427
    # PCC_F21 = 0.20195486429974477
    # PCC_F23 = 0.4498487034369941
    # PCC_F31 = 0.042351225660531684
    # PCC_F32 = 0.025337460695713448

    CC_count1, CC_count2, CC_count3 = 0, 0, 0
    # CC
    for i in range(len(prob_train)):
        if (np.argmax(prob_train[i])) == 0:
            CC_count1 += 1
        if (np.argmax(prob_train[i])) == 1:
            CC_count2 += 1
        if (np.argmax(prob_train[i])) == 2:
            CC_count3 += 1
    PCC_count1, PCC_count2, PCC_count3 = 0, 0, 0
    # PCC
    for i in range(len(prob_train)):
        PCC_count1 += prob_train[i][0]
        PCC_count2 += prob_train[i][1]
        PCC_count3 += prob_train[i][2]

    # calculate quantifications for each of the methods
    CC_share1 = CC_count1 / (CC_count1 + CC_count2 + CC_count3)
    CC_share2 = CC_count2 / (CC_count1 + CC_count2 + CC_count3)
    CC_share3 = CC_count3 / (CC_count1 + CC_count2 + CC_count3)

    PCC_share1 = PCC_count1 / (PCC_count1 + PCC_count2 + PCC_count3)
    PCC_share2 = PCC_count2 / (PCC_count1 + PCC_count2 + PCC_count3)
    PCC_share3 = PCC_count3 / (PCC_count1 + PCC_count2 + PCC_count3)

    ACC_share1 = (CC_share1 - ((CC_F12 - CC_F13) * (CC_share2 - CC_F23)) / (CC_T2 - CC_F23 + 0.0001) - CC_F13 + 0.0001) / \
                 (CC_T1 - ((CC_F12 - CC_F13) * (CC_F21 - CC_F23)) / (CC_T2 - CC_F23 + 0.0001) - CC_F13 + 0.0001)
    ACC_share2 = (CC_share2 - ((CC_F21 - CC_F23) * (CC_share1 - CC_F13)) / (CC_T1 - CC_F13 + 0.0001) - CC_F23 + 0.0001) / \
                 (CC_T2 - ((CC_F21 - CC_F23) * (CC_F12 - CC_F13)) / (CC_T1 - CC_F13 + 0.0001) - CC_F23 + 0.0001)
    ACC_share3 = 1 - ACC_share1 - ACC_share2

    PACC_share1 = (PCC_share1 - ((PCC_F12 - PCC_F13) * (PCC_share2 - PCC_F23)) / (PCC_T2 - PCC_F23) - PCC_F13 + 0.0001) / \
                  (PCC_T1 - ((PCC_F12 - PCC_F13) * (PCC_F21 - PCC_F23)) / (PCC_T2 - PCC_F23) - PCC_F13 + 0.0001)
    PACC_share2 = (PCC_share2 - ((PCC_F21 - PCC_F23) * (PCC_share1 - PCC_F13)) / (PCC_T1 - PCC_F13) - PCC_F23 + 0.0001) / \
                  (PCC_T2 - ((PCC_F21 - PCC_F23) * (PCC_F12 - PCC_F13)) / (PCC_T1 - PCC_F13) - PCC_F23 + 0.0001)
    PACC_share3 = 1 - PACC_share1 - PACC_share2
    return [CC_T1, CC_T2, CC_T3, CC_F12, CC_F13, CC_F21, CC_F23, CC_F31, CC_F31, CC_share1, CC_share2, CC_share3, PCC_share1, PCC_share2, PCC_share3, ACC_share1, ACC_share2, ACC_share3, PACC_share1, PACC_share2, PACC_share3]