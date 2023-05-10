import logging
from typing import Optional

import numpy as np

from .loader import ConnDataset


class KFoldGroup:
    def __init__(
            self,
            dataset: ConnDataset,
            log: logging.Logger,
            session: int,
            batch_size: int,
            fold: int, seed: int,
            stratified: bool = False
    ):
        self.dataset = dataset
        self.log = log
        self.fold = fold
        self.seed = seed
        self.stratified = stratified
        self.n_session = session
        self.batch_size = batch_size

        self.n_sample_orig = dataset.n_sample_orig
        self.n_sample_noise = dataset.n_sample_noise
        self.n_sample_mixup = dataset.n_sample_mixup
        self.n_sample = dataset.n_sample

        self.__check_index()

    def __check_index(self):
        if self.fold <= 2 \
                or self.fold > self.n_sample \
                or int(self.n_sample % self.fold) != 0:
            self.log.error(f'Incorrect k fold config: {self.n_sample} divided by {self.fold}')
            raise RuntimeError

    def __check_stratified(self):
        if self.batch_size % self.n_session != 0 \
                or self.batch_size * self.fold != self.n_sample_orig + self.n_sample_noise:
            self.log.error(f'Incorrect stratified k fold config: {self.n_sample} divided by {self.fold}')
            raise RuntimeError

    def __create_seq_mask(self, ids: np.ndarray, start: int, stop: int):
        mask = np.zeros(self.n_sample, dtype=bool)
        mask[ids[start:stop]] = True
        return mask

    def __create_mixup_mask(self):
        mask = np.zeros(self.n_sample, dtype=bool)
        if self.n_sample_mixup > 0:
            mask[-self.n_sample_mixup:] = True
        return mask

    def __group_random_mask(self):
        n_sample_train = self.n_sample_orig + self.n_sample_noise
        mix_mask = self.__create_mixup_mask()

        ids: Optional[np.ndarray] = np.arange(0, n_sample_train)
        np.random.RandomState(self.seed).shuffle(ids)
        fold_sizes = np.full(self.fold, n_sample_train // self.fold, dtype=int)
        fold_sizes[: n_sample_train % self.fold] += 1

        test_mask = self.__create_seq_mask(ids, n_sample_train - fold_sizes[-1], n_sample_train)
        # except last group as test set
        fold_sizes = np.delete(fold_sizes, -1)

        cur = 0
        for step in fold_sizes:
            start, stop = cur, cur + step
            valid_mask = self.__create_seq_mask(ids, start, stop)
            train_mask = np.logical_not(
                np.logical_or(
                    mix_mask,
                    np.logical_or(valid_mask, test_mask)
                )
            )

            yield train_mask, valid_mask, test_mask
            cur = stop

    def __create_sub_mask(self, ids: np.ndarray, subs: np.ndarray, start: int, stop: int):
        mask = np.zeros(self.n_sample, dtype=bool)
        index = []
        for i in range(start, stop, 1):
            sub_id = subs[i]
            for j in range(sub_id * self.n_session, (sub_id + 1) * self.n_session):
                index.append(j)
        mask[ids[index]] = True
        return mask

    def __group_stratified_mask(self):
        self.__check_stratified()

        n_sample_train = self.n_sample_orig + self.n_sample_noise
        n_subject = n_sample_train // self.n_session
        n_step = self.batch_size // self.n_session
        mix_mask = self.__create_mixup_mask()

        perm = np.random.RandomState(self.seed).permutation(n_subject)
        ids: Optional[np.ndarray] = np.arange(0, n_sample_train)
        test_mask = self.__create_sub_mask(ids, perm, n_subject - n_step, n_subject)

        for start in range(0, n_subject - n_step, n_step):
            stop = start + n_step
            valid_mask = self.__create_sub_mask(ids, perm, start, stop)
            train_mask = np.logical_not(
                np.logical_or(
                    mix_mask,
                    np.logical_or(valid_mask, test_mask)
                )
            )

            yield train_mask, valid_mask, test_mask

    def __group_mask(self):
        if self.stratified:
            return self.__group_stratified_mask()
        else:
            return self.__group_random_mask()

    def split(self):
        index = np.arange(0, self.n_sample)
        for train_mask, valid_mask, test_mask in self.__group_mask():
            train_index: np.ndarray = index[train_mask]
            valid_index = index[valid_mask]
            test_index = index[test_mask]

            self.log.info(f'Train set index array: {train_index.tolist()}')
            self.log.info(f'Validate set index array: {valid_index.tolist()}')
            self.log.info(f'Test set index array: {test_index.tolist()}')

            yield self.dataset[train_index], \
                self.dataset[valid_index], \
                self.dataset[test_index]
