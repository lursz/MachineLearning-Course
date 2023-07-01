from base import Heuristic
from problems.grid_pathfinding.grid_pathfinding import GridPathfinding
from problems.grid_pathfinding.grid import GridCoord


class GridManhattanHeuristic(Heuristic[GridCoord]):
 
    def __init__(self, problem: GridPathfinding):
        self.problem = problem

    def __call__(self, state: GridCoord) -> float:
        # !DONE:
        # Calculate a manhattan distance:
        # - 'state' is the current state 
        
        D = 1   # straight_foward_weight
        dx = abs(state.x - self.problem.goal.x)
        dy = abs(state.y - self.problem.goal.y)
        return (dx + dy) * D
        raise NotImplementedError

