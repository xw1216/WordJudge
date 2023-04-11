import time


class Timer(object):
    def __init__(self):
        self.__time_start = .0
        self.__time_end = .0
        self.__time_acc = .0
        self.__duration = .0
        self.__count = 0

    def start(self):
        self.__time_start = time.time()
        self.__time_end = .0
        self.__duration = .0

    def end(self):
        self.__time_end = time.time()
        self.__duration = self.__time_end - self.__time_start
        self.__time_acc += self.__duration
        self.__count += 1

    def last(self):
        return self.__duration

    @staticmethod
    def to_datetime(t: float):
        t_arr = time.localtime(t)
        datetime = time.strftime("%m-%d %H:%M:%S", t_arr)
        return datetime

    def __repr__(self) -> str:
        return str(type(Timer)) + \
            f'Start: {Timer.to_datetime(self.__time_start)}, ' \
            f'Duration: {Timer.to_datetime(self.__duration)}'
