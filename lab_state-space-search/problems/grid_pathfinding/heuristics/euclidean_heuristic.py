from base import Heuristic
from problems.grid_pathfinding.grid_pathfinding import GridPathfinding
from problems.grid_pathfinding.grid import GridCoord
from math import sqrt


class GridEuclideanHeuristic(Heuristic[GridCoord]):
 
    def __init__(self, problem: GridPathfinding):
        self.problem = problem

    def __call__(self, state: GridCoord) -> float:
        # !DONE:
        # Calculate an euclidean distance:
        # - 'state' is the current state 
        # - 'self.problem.goal' is the goal state
        D = 1   # straight_foward_weight
        dx = abs(state.x - self.problem.goal.x)
        dy = abs(state.y - self.problem.goal.y)
        return D * sqrt(dx * dx + dy * dy)
        raise NotImplementedError
  
