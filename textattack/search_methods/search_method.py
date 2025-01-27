"""
Search Method Abstract Class
===============================
"""


from abc import ABC, abstractmethod

from textattack.shared.utils import ReprMixin


class SearchMethod(ReprMixin, ABC):
    """This is an abstract class that contains main helper functionality for
    search methods.

    A search method is a strategy for applying transformations until the
    goal is met or the search is exhausted.
    """

    def __init__(self, search_all=False, sort_results=False):
        self.search_all = search_all
        self.sort_results = sort_results

    def __call__(self, initial_result):
        """Ensures access to necessary functions, then calls
        ``perform_search``"""
        if not hasattr(self, "get_transformations"):
            raise AttributeError(
                "Search Method must have access to get_transformations method"
            )
        if not hasattr(self, "get_goal_results"):
            raise AttributeError(
                "Search Method must have access to get_goal_results method"
            )
        if not hasattr(self, "filter_transformations"):
            raise AttributeError(
                "Search Method must have access to filter_transformations method"
            )

        results = self.perform_search(initial_result)
        # ensure that the number of queries for this GoalFunctionResult is up-to-date
        # results.num_queries = self.goal_function.num_queries
        return results, self.goal_function.num_queries

    @abstractmethod
    def perform_search(self, initial_result):
        """Perturbs `attacked_text` from ``initial_result`` until goal is
        reached or search is exhausted.

        Must be overridden by specific search methods.
        """
        raise NotImplementedError()

    def check_transformation_compatibility(self, transformation):
        """Determines whether this search method is compatible with
        ``transformation``."""
        return True

    @property
    def is_black_box(self):
        """Returns `True` if search method does not require access to victim
        model's internal states."""
        raise NotImplementedError()

    def get_victim_model(self):
        if self.is_black_box:
            raise NotImplementedError(
                "Cannot access victim model if search method is a black-box method."
            )
        else:
            return self.goal_function.model
