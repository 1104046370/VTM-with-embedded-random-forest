import pickle
import numpy as np


def pred(model, inp):
    y_ns = model.predict_proba(inp)
    k = np.max(y_ns[0])
    if k < 0.5:
        return 6
    else:
        ret = model.classes_[np.argmax(y_ns[0])]
        return int(ret)

def main(cuw, cuh, *args):  # uorg表示实际像素，upred表示预测像素
    # cuw = cu

    inp = np.array([[*args]])

    with open('./model/' + str(cuw) + '_' + str(cuh) + '.pkl', "rb") as f:
        m = pickle.load(f)
    ret = pred(m, inp)

    return ret

if __name__ == '__main__':
    pass
