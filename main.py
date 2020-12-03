from typing import List, Dict, Tuple, Any
import copy
from abc import *
import math

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


class ProblemExample_4_1_TLBR(IProblemSolving):
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


class ProblemExample_4_2_Time(IProblemSolving):
    def solve(self, input_data: List[int]):
        n = input_data[0]
        # 00시 00분 00초부터 N시 59분 59초까지 중, 3을 하나라도 포함하는 모든 경우의 수
        #   n : 0~23
        count: int = 0
        for h in range(n+1):
            for m in range(60):
                for s in range(60):
                    if '3' in f'{h}:{m}:{s}':
                        count += 1
        return count


class Problem_2_RoyalNight(IProblemSolving):
    def solve(self, input_data: List[str]):
        # 이동할 수 있는 좌표의 경우의 수
        available = [[2, 1], [2, -1], [-2, 1], [-2, -1],
                     [1, 2], [1, -2], [-1, 2], [-1, -2]]
        loc: str = input_data[0] # a1 형태
        m: Dict[str, int] = {'a': 1, 'b': 2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8}

        row: int = m[loc[0]]
        col: int = int(loc[1])
        count = 0
        for r, c in available:
            tr = row + r
            tc = col + c
            if 1 <= tr <= 8 and 1 <= tc <= 8:
                count += 1
        return count


class Problem_3_GameDevelopment(IProblemSolving):
    def solve(self, input_data: List[List[int]]):
        n, m = input_data[0]
        row, col, direction = input_data[1]
        # 1은 바다, 0은 육지
        # 캐릭터 방향이 0이면 북쪽, 1이면 동쪽, 2면 남쪽, 3이면 서쪽

        area = list(map(lambda x: list(map(lambda y: y, x)), input_data[2:2+n]))
        visit = list(map(lambda x: list(map(lambda y: 0, range(m))), range(n)))
        count = 0
        done_flag = False

        visit[row][col] = 1     # 처음 위치한 곳은 가본 곳

        def is_movable(r, c) -> bool:
            if 0 <= r < n and 0 <= c < m:
                if area[r][c] == 0:
                    return True
            return False

        def get_next_field_pos(cur_row, cur_col, cur_direction):
            direction_area = {0: [-1, 0], 3: [0, -1], 2: [1, 0], 1: [0, 1]}
            return [cur_row + direction_area[cur_direction][0], cur_col + direction_area[cur_direction][1]]

        def get_next_direction(cur_direction):
            direction_policy = {0: 3, 3: 2, 2: 1, 1: 0}
            return direction_policy[cur_direction]

        def is_visited(r, c) -> bool:
            if visit[r][c] == 1:
                return True
            return False

        def move_back(r, c, d) -> Tuple[int, int]:
            move_back_policy = {0: [1, 0], 1: [0, -1], 2: [-1, 0], 3: [0, 1]}
            r += move_back_policy[d][0]
            c += move_back_policy[d][1]
            return r, c

        check_count: int = 0
        while done_flag is False:
            direction = get_next_direction(direction)
            next_row, next_col = get_next_field_pos(row, col, direction)        # 갈 곳을 정함.
            if is_movable(next_row, next_col) is False or is_visited(next_row, next_col) is True:
                # 정한 갈 곳으로 갈 수 없다면 다시 처음으로 돌아간다.
                # 대신 내부 카운트 값은 증가시킨다. 이게 4가 되면 4번 살펴봤다는 의미라 3번 루틴으로 가야 한다.
                check_count += 1
                if check_count >= 4:
                    # 만약 네 칸 모두 가본 칸이거나 바다로 되어 있다면, 바라보는 방향 유지한 채 한 칸 뒤로 간다.
                    check_count = 0
                    row, col = move_back(row, col, direction)
                    if area[row][col] == 1:
                        # 바다면 움직임을 멈춘다.
                        done_flag = True
                        break
                    else:
                        # 육지면 가본곳을 추가한다.
                        visit[row][col] = 1
                continue
            else:
                # 가보지 않은 칸으로 진행한다.
                row = next_row
                col = next_col
                visit[row][col] = 1
                check_count = 0

        # 방문한 칸 수를 출력
        visited_count = 0
        from functools import reduce
        for rows in visit:
            visited_count += reduce(lambda x, y: x + y, rows)
        return visited_count


if __name__ == '__main__':
    problems = [
        ProblemExample1([[[5,8,3], [2,4,5,4,6]]]),
        Problem3NumCardGame([
            [[3, 3], [3, 1, 2], [4, 1, 4], [2, 2, 2]],
            [[2, 4], [7, 3, 1, 8], [3, 3, 3, 4]]
        ]),
        Problem4Until1([[25, 5]]),
        ProblemExample_4_1_TLBR([[[5], ['R', 'R', 'R', 'U', 'D', 'D']]]),
        ProblemExample_4_2_Time([[5]]),
        Problem_2_RoyalNight([['a1']]),
        Problem_3_GameDevelopment([[[4,4], [1,1,0], [1,1,1,1], [1,0,0,1], [1,1,0,1], [1,1,1,1]]]),
    ]

    for p in problems:
        p.run()
