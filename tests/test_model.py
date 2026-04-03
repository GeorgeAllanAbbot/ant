import unittest
from unittest.mock import MagicMock
import sys

# Mock numpy before importing any SDK modules that depend on it
sys.modules['numpy'] = MagicMock()

from SDK.backend.model import Ant, NO_MOVE
from SDK.utils.constants import OFFSET, AntKind

class TestAntRecordMove(unittest.TestCase):
    def setUp(self):
        # Initializing an Ant at (10, 10), player 0, hp 20, level 0
        self.ant = Ant(ant_id=1, player=0, x=10, y=10, hp=20, level=0, kind=AntKind.WORKER)

    def test_record_no_move(self):
        initial_x, initial_y = self.ant.x, self.ant.y
        initial_path_len = self.ant.path_len_total
        initial_trail_len = len(self.ant.trail_cells)

        self.ant.record_move(NO_MOVE)

        self.assertEqual(self.ant.x, initial_x)
        self.assertEqual(self.ant.y, initial_y)
        self.assertEqual(self.ant.path_len_total, initial_path_len + 1)
        self.assertEqual(self.ant.last_move, NO_MOVE)
        self.assertEqual(len(self.ant.trail_cells), initial_trail_len)

    def test_record_move_even_y(self):
        # (10, 10) -> 10 is even
        self.assertEqual(self.ant.y % 2, 0)

        for direction in range(6):
            ant = Ant(ant_id=1, player=0, x=10, y=10, hp=20, level=0, kind=AntKind.WORKER)
            dx, dy = OFFSET[0][direction]

            ant.record_move(direction)

            self.assertEqual(ant.x, 10 + dx)
            self.assertEqual(ant.y, 10 + dy)
            self.assertEqual(ant.last_move, direction)
            self.assertEqual(ant.path_len_total, 1)
            self.assertEqual(ant.trail_cells[-1], (ant.x, ant.y))
            self.assertEqual(len(ant.trail_cells), 2) # [(10, 10), (new_x, new_y)]

    def test_record_move_odd_y(self):
        # Testing movement from an odd y coordinate (11)
        for direction in range(6):
            ant = Ant(ant_id=1, player=0, x=10, y=11, hp=20, level=0, kind=AntKind.WORKER)
            self.assertEqual(ant.y % 2, 1)
            dx, dy = OFFSET[1][direction]

            ant.record_move(direction)

            self.assertEqual(ant.x, 10 + dx)
            self.assertEqual(ant.y, 11 + dy)
            self.assertEqual(ant.last_move, direction)
            self.assertEqual(ant.path_len_total, 1)
            self.assertEqual(ant.trail_cells[-1], (ant.x, ant.y))
            self.assertEqual(len(ant.trail_cells), 2)

    def test_multiple_moves(self):
        # Test a sequence of moves
        initial_x, initial_y = 10, 10
        ant = Ant(ant_id=1, player=0, x=initial_x, y=initial_y, hp=20, level=0, kind=AntKind.WORKER)

        # Move 0 (dx=0, dy=1 if even y) -> (10, 11)
        ant.record_move(0)
        self.assertEqual((ant.x, ant.y), (10, 11))

        # Move 1 (dx=-1, dy=0 if odd y) -> (9, 11)
        ant.record_move(1)
        self.assertEqual((ant.x, ant.y), (9, 11))

        # Move 2 (dx=-1, dy=-1 if odd y) -> (8, 10)
        ant.record_move(2)
        self.assertEqual((ant.x, ant.y), (8, 10))

        self.assertEqual(ant.path_len_total, 3)
        self.assertEqual(len(ant.trail_cells), 4) # initial + 3 moves
        self.assertEqual(ant.trail_cells, [(10, 10), (10, 11), (9, 11), (8, 10)])

if __name__ == '__main__':
    unittest.main()
