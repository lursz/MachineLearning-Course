from abc import ABC
from typing import List

from local_search.helpers.camel_to_snake import camel_to_snake
from local_search.problems.base.problem import Goal
from local_search.problems.graph_coloring_problem.models.edge import Edge
from local_search.problems.graph_coloring_problem.state import GraphColoringState


class GraphColoringGoal(Goal, ABC):
    """
    Base class for goals of the graph coloring problem
    """
    goals = {}

    def __init__(self, edges: List[Edge], n_vertices: int):
        self.edges = edges
        self.n_vertices = n_vertices

    def __init_subclass__(cls):
        GraphColoringGoal.goals[camel_to_snake(cls.__name__)] = cls

    def _num_colors(self, state: GraphColoringState) -> int:
        # !DONE:
        # return number of distinct colors used in `state.coloring`
        distinct_colors = set()
        for vertex in state.coloring:
            distinct_colors.add(vertex.color)
        return len(distinct_colors)



    def _bad_edges(self, state: GraphColoringState) -> List[int]:
        # !DONE:
        # return number of bad edges of every color class in the graph
        # tip 1. `self.edges` is the list of 'Edge' in the graph
        # tip 2. example usage: `state.coloring[edge.start].color` 
        #        is color of the edge start in the current state
        number_of_bad_edges = [0] * self.n_vertices
 
        for edge in self.edges:
            if state.coloring[edge.start].color == state.coloring[edge.end].color:
                number_of_bad_edges[state.coloring[edge.start].color] += 1
        return number_of_bad_edges
        

    def _color_classes(self, state: GraphColoringState) -> List[int]:
        # !DONE:
        # return sizes of the color classes
        # - assume the worst case — there is as many colors as there are vertices
        #   so the result should be a list of size `self.n_vertices`
        # - the result is a list with values corresponding to sizes of the color classes, e.g.
        #   `result[0] = 5` means that there five nodes in `state` with `color = 0`
        array_of_colours = [0] * self.n_vertices

        for vertex in state.coloring:
            array_of_colours[vertex.color] += 1
        return array_of_colours
    


    def human_readable_objective_for(self, state: GraphColoringState) -> str:
        return f"{self._num_colors(state)} colors"
