from turtle import distance
from base import state
from problems.n_puzzle import NPuzzleState

from problems.n_puzzle.heuristics.n_puzzle_abstract_heuristic import NPuzzleAbstractHeuristic


class NPuzzleManhattanHeuristic(NPuzzleAbstractHeuristic):

    def __call__(self, state: NPuzzleState) -> float:
        # !DONE: implement manhattan distance heuristic
        # Calculate a manhattan distance between tiles and their expected places
        # The result should be sum of those distances. 
        # tip 1.'state' is the current state, 
        # tip 2. you can use self.positions function to get from it a dictionary:
        #   { tile_number : (x_coordinate, y_coordinate) }
        # tip 3. self.goal_coords contains such a dictionary for the goal state
        
        distances = []
        for tile in self.positions(state):
            distance = abs(self.positions(state)[tile][0] - self.goal_coords[tile][0]) + abs(self.positions(state)[tile][1] - self.goal_coords[tile][1])
            distances.append(distance)
        return sum(distances)