from typing import List, Dict, Tuple, Any
import copy
from abc import *


class IProblemSolving(metaclass=ABCMeta):
    input_data_list: List[List[Any]]

    def __init__(self, input_data_list: List[Any]):
        self.set_params(input_data_list)

    @abstractmethod
    def solve(self, input_data: List[Any]):
        pass

    def set_params(self, input_data_list: List[Any]):
        self.input_data_list = input_data_list

    def run(self):
        for input_data in self.input_data_list:
            print(f'{self.__class__.__name__} >> input_data: {input_data} result: {self.solve(input_data)}')


class ProblemExample1(IProblemSolving):
    def solve(self, input_data: List[List[int]]) -> int:
        n, m, k = input_data[0]
        result = 0
        tmp: List[int] = copy.copy(input_data[1])
        tmp.sort(reverse=True)
        for i in range(1, m+1):
            if i % (k+1) != 0:
                result += tmp[0]
            else:
                result += tmp[1]
        return result


class Problem3NumCardGame(IProblemSolving):
    def solve(self, input_data: List[List[int]]) -> int:
        n, m = input_data[0]
        r = list(map(lambda x: min(x), input_data[1:n+1]))
        return max(r)


class Problem4Until1(IProblemSolving):
    def solve(self, input_data: List[int]) -> int:
        n, k = input_data
        count = 0
        while n > 1:
            if n % k == 0:
                n = n / k
            else:
                n -= 1
            count += 1
        return count


class Problem_4_1_TLBR(IProblemSolving):
    def solve(self, input_data: List[List[Any]]) -> List[int]:
        pos = [1, 1]
        mat_size = input_data[0][0]

        for order in input_data[1]:
            if order == 'R' and pos[1] < mat_size:
                pos[1] += 1
            elif order == 'L' and pos[1] > 1:
                pos[1] -= 1
            elif order == 'U' and pos[0] > 1:
                pos[0] -= 1
            elif order == 'D' and pos[0] < mat_size:
                pos[0] += 1
        return pos


if __name__ == '__main__':
    problems = [
        ProblemExample1([[[5,8,3], [2,4,5,4,6]]]),
        Problem3NumCardGame([
            [[3, 3], [3, 1, 2], [4, 1, 4], [2, 2, 2]],
            [[2, 4], [7, 3, 1, 8], [3, 3, 3, 4]]
        ]),
        Problem4Until1([[25, 5]]),
        Problem_4_1_TLBR([[[5], ['R', 'R', 'R', 'U', 'D', 'D']]]),
    ]

    for p in problems:
        p.run()
