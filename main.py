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


class Problem_3_DFS_BFS_FrozenBeverage(IProblemSolving):
    def solve(self, input_data: List[List[Any]]):
        n = input_data[0][0]
        m = input_data[0][1]
        slot = []
        for idx, row in enumerate(input_data[1:]):
            slot.append([])
            for val in row:
                slot[idx].append(int(val[0]))

        # 왼쪽 위부터 모든 항목을 확인하며 값을 바꿔나감.
        def fill_empty(r, c):
            slot[r][c] = 1
            if r + 1 < len(slot) and slot[r+1][c] == 0:
                fill_empty(r + 1, c)
            if c + 1 < len(slot[r]) and slot[r][c+1] == 0:
                fill_empty(r, c + 1)
            if r - 1 >= 0 and slot[r-1][c] == 0:
                fill_empty(r-1, c)
            if c - 1 >= 0 and slot[r][c-1] == 0:
                fill_empty(r, c-1)

        ice_count = 0
        for row_idx, row in enumerate(slot):
            for col_idx, el in enumerate(row):
                if el == 0:
                    fill_empty(row_idx, col_idx)
                    ice_count += 1
        return ice_count


class Problem_4_DFS_BFS_EscapeMaze(IProblemSolving):
    def solve(self, input_data: List[Any]):
        n, m = input_data[0]
        maze = []
        for row in range(n):
            maze.append(list(map(int, input_data[1+row])))
        # 1은 갈 수 있음. 0은 갈 수 없음.
        # 움직여야 하는 최소 칸 수. BFS.
        # 최초 위치 0,0, 탈출구는 n-1, m-1
        from collections import deque
        queue = deque()
        queue.append((0, 0, 0))

        def get_min_move(q: deque) -> int:
            r, c, moves = q.popleft()
            if r == n-1 and c == m-1:
                return moves
            if r + 1 < n and maze[r+1][c] == 1:
                q.append((r+1, c, moves+1))
            elif r - 1 >= 0 and maze[r-1][c] == 1:
                q.append((r-1, c, moves+1))
            elif c + 1 < m and maze[r][c+1] == 1:
                q.append((r, c + 1, moves + 1))
            elif c - 1 >= 0 and maze[r][c-1] == 1:
                q.append((r, c - 1, moves + 1))
            return get_min_move(q)
        r = get_min_move(queue)
        return r

        pass


class Problem_Test_Sort_ComparingInsertionAndQuickSortWhenAlmostSorted(IProblemSolving):
    def solve(self, input_data: List[Any]):
        # input_data는 따로 사용하지 않는다.
        rarely_shuffled: List[int] = []
        elements: int = 1000000
        shuffle_probabilities: List[float] = [0.5, 0.3, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001] # 10000, 1000, 100, 10

        def make_rare_shuffled_array(elements_cnt: int, shuffle_probability: float) -> List[int]:
            ascend_arr: List[int] = [i for i in range(elements_cnt)]
            r: List[int] = []
            for i in ascend_arr:
                import random
                if random.random() < shuffle_probability:
                    pass
                    rarely_shuffled.insert(random.randint(0, len(rarely_shuffled)), i)
                else:
                    rarely_shuffled.append(i)
            return r

        # insertion sort
        def insertion_sort(arr: List[int]) -> List[int]:
            for i in range(1, len(arr)):
                for j in range(i, 0, -1):   # i부터 1까지 -1씩 감소
                    if arr[j] < arr[j-1]:   # 한칸씩 왼쪽으로 이동
                        arr[j], arr[j-1] = arr[j-1], arr[j]     # swap
                    else:   # 이미 정렬되어 있는 배열을 만나면 그 위치에서 멈춤
                        break
            return arr

        # quick sort
        def quick_sort(arr: List[int], start: int, end: int) -> List[int]:
            if start >= end:    # 원소가 한 개면 종료
                return arr
            pivot = start   # pivot은 첫 번째 원소
            left = start + 1
            right = end
            while left <= right:
                # pivot보다 큰 데이터 찾을 때까지 반복
                while left <= end and arr[left] <= arr[pivot]:
                    left += 1
                # pivot보다 작은 데이터 찾을 떄까지 반복
                while right > start and arr[right] >= arr[pivot]:
                    right += 1
                if left > right:    # 엇갈렸다면 작은 데이터와 피벗을 교체
                    arr[right], arr[pivot] = arr[pivot], arr[right]
                else:   # 엇갈리지 않았다면 작은 데이터와 큰 데이터를 교체
                    arr[left], arr[right] = arr[right], arr[left]
            # 분할 이후 왼쪽과 오른쪽에 각각 정렬 수행
            arr = quick_sort(arr, start, right - 1)
            arr = quick_sort(arr, right+1, end)
            return arr

        import time
        import copy
        for sp in shuffle_probabilities:
            print(f'processing el: {elements} sp: {sp}')
            to_be_sorted: List[int] = make_rare_shuffled_array(elements, sp)
            arr_a = copy.copy(to_be_sorted)
            arr_b = copy.copy(to_be_sorted)
            start_a = time.time()
            insertion_sort(arr_a)
            end_a = time.time()
            start_b = time.time()
            quick_sort(arr_b, 0, len(arr_b))
            end_b = time.time()
            winner = 'insertion' if (end_b-start_b) > (end_a-start_a) else 'quick'
            print(f'elements: {elements} probability: {sp} insertion: {end_a-start_a:.6f}s '
                  f'quick: {end_b-start_b:.6f}s winner: {winner}')


def code_practice():
    s = [3,5,7,9,11,13]
    print(s[1::2])
    from collections import deque
    queue = deque()
    graph = [[0 for _ in range(3)] for _ in range(3)]
    print(graph)



if __name__ == '__main__':
    problems = [
        # ProblemExample1([[[5,8,3], [2,4,5,4,6]]]),
        # Problem3NumCardGame([
        #     [[3, 3], [3, 1, 2], [4, 1, 4], [2, 2, 2]],
        #     [[2, 4], [7, 3, 1, 8], [3, 3, 3, 4]]
        # ]),
        # Problem4Until1([[25, 5]]),
        # ProblemExample_4_1_TLBR([[[5], ['R', 'R', 'R', 'U', 'D', 'D']]]),
        # ProblemExample_4_2_Time([[5]]),
        # Problem_2_RoyalNight([['a1']]),
        # Problem_3_GameDevelopment([[[4,4], [1,1,0], [1,1,1,1], [1,0,0,1], [1,1,0,1], [1,1,1,1]]]),
        # Problem_3_DFS_BFS_FrozenBeverage([[[15, 14],
        #                                    '00000111100000',
        #                                    '11111101111110',
        #                                    '11011101101110',
        #                                    '11011101100000',
        #                                    '11011111111111',
        #                                    '11011111111100',
        #                                    '11000000011111',
        #                                    '01111111111111',
        #                                    '00000000011111',
        #                                    '01111111111000',
        #                                    '00011111111000',
        #                                    '00000001111000',
        #                                    '11111111110011',
        #                                    '11100011111111',
        #                                    '11100011111111'
        #                                    ]]),
        # Problem_4_DFS_BFS_EscapeMaze([[[5, 6], '101010', '111111', '000001', '111111', '111111']])
        Problem_Test_Sort_ComparingInsertionAndQuickSortWhenAlmostSorted([[]])
    ]

    # code_practice()

    for p in problems:
        p.run()
