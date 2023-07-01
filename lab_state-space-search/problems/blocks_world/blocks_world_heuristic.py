


from typing import Dict, List
from base.heuristic import Heuristic
from problems.blocks_world.blocks_world_problem import BlocksWorldProblem, BlocksWorldState

class BlocksWorldNaiveHeuristic(Heuristic):

    def __init__(self, problem: BlocksWorldProblem) -> None:
        super().__init__(problem)
        self.expected_columns = self._calculate_expected_columns(problem.goal)
        self.expected_fundaments = self._calculate_expected_fundaments(problem.goal)

    def _calculate_expected_columns(self, goal: BlocksWorldState) -> Dict[str, int]:
        # !DONE:
        # return a dict of form:
        # { <block name> : <index of column in the goal state> }
        dict = {}
        for i, cols in enumerate(goal.columns):
            for block in cols:
                dict[block] = i
        return dict
    
    def _calculate_expected_fundaments(self, goal: BlocksWorldState) -> Dict[str, List[str]]:
        # !DONE:
        # return a dict of form:
        # { <block name> : <list of the blocks below it in the goal state> }
        dict = {}
        for i, cols in enumerate(goal.columns):
            for j, block in enumerate(cols):
                dict[block] = cols[:j]
        return dict
                

    def __call__(self, state: BlocksWorldState) -> int:
        # !DONE:
        # - add `1` to the heuristic value per each block placed in an incorrect column
        # - for other blocks, add `2` if their fundament is incorrect 
        # tip. use self.expected_columns and self.expected_fundaments
        heuristic_value = 0
        for i, column in enumerate(state.columns):
            for j, block in enumerate(column):
                if self.expected_columns[block] != i:
                    heuristic_value += 1
                elif self.expected_fundaments[block] != column[:j]:
                    heuristic_value += 2
        return heuristic_value
