import numpy as np


class DataInput:
    def __init__(self, data, batch_size):

        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def next(self):

        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size: min((self.i + 1) * self.batch_size,
                                                     len(self.data))]
        self.i += 1

        y, user_feature, item_feature, cate, keyword, keyword2, tag1, tag2, tag3, ks1, ks2, sl = \
            [], [], [], [], [], [], [], [], [], [], [], []
        for t in ts:
            y.append(t[0])
            user_feature.append(t[1])
            item_feature.append(t[2])
            cate.append((t[3]))
            keyword.append(t[4])
            keyword2.append(t[5])
            tag1.append(t[6])
            tag2.append(t[7])
            tag3.append(t[8])
            ks1.append(t[9])
            ks2.append(t[10])

            sl.append(len(t[12]))
        max_sl = max(sl)

        hist_cate = np.zeros([len(ts), max_sl], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[11])):
                hist_cate[k][l] = t[11][l]
            k += 1

        hist_keyword = np.zeros([len(ts), max_sl], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[12])):
                hist_keyword[k][l] = t[12][l]
            k += 1

        hist_keyword2 = np.zeros([len(ts), max_sl], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[13])):
                hist_keyword2[k][l] = t[13][l]
            k += 1

        hist_tag1 = np.zeros([len(ts), max_sl], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[14])):
                hist_tag1[k][l] = t[14][l]
            k += 1

        hist_tag2 = np.zeros([len(ts), max_sl], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[15])):
                hist_tag2[k][l] = t[15][l]
            k += 1

        hist_tag3 = np.zeros([len(ts), max_sl], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[16])):
                hist_tag3[k][l] = t[16][l]
            k += 1

        hist_ks1 = np.zeros([len(ts), max_sl], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[17])):
                hist_ks1[k][l] = t[17][l]
            k += 1

        hist_ks2 = np.zeros([len(ts), max_sl], np.int64)
        k = 0
        for t in ts:
            for l in range(len(t[18])):
                hist_ks2[k][l] = t[18][l]
            k += 1

        return self.i, (y, user_feature, item_feature, cate, keyword, keyword2, tag1, tag2, tag3, ks1, ks2,
                        hist_cate, hist_keyword, hist_keyword2, hist_tag1, hist_tag2, hist_tag3, hist_ks1, hist_ks2, sl)
