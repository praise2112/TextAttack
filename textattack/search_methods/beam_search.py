"""
Beam Search
===============

"""
import numpy as np
import math

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod


class BeamSearch(SearchMethod):
    """An attack that maintinas a beam of the `beam_width` highest scoring
    AttackedTexts, greedily updating the beam with the highest scoring
    transformations from the current beam.

    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        transformation: The type of transformation.
        beam_width (int): the number of candidates to retain at each step
    """

    def __init__(self, beam_width=8, **kwargs):
        super().__init__(**kwargs)
        self.beam_width = beam_width

    def remove_dup(self, results):
        new_results = []
        for result in results:
            skip = any(result.attacked_text.text == uniq_res.attacked_text.text for uniq_res in new_results)
            if skip:
                continue
            new_results.append(result)
        return new_results

    def perform_search(self, initial_result):
        beam = [initial_result.attacked_text]
        best_results = []
        search_over = False
        while not search_over:
        # while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            potential_next_beam = []
            for text in beam:
                transformations = self.get_transformations(
                    text, original_text=initial_result.attacked_text
                )
                potential_next_beam += transformations

            if len(potential_next_beam) == 0:
                # If we did not find any possible perturbations, give up.
                return best_results or [initial_result]
            results, search_over = self.get_goal_results(potential_next_beam)
            scores = np.array([r.score for r in results])
            if self.search_all:
                best_results.extend(
                    list(filter(lambda x: x.goal_status == GoalFunctionResultStatus.SUCCEEDED, results)))
            else:
                best_results = [results[scores.argmax()]]
                if best_results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    return best_results

            # Refill the beam. This works by sorting the scores
            # in descending order and filling the beam from there.
            best_indices = (-scores).argsort()[: self.beam_width]
            beam = [potential_next_beam[i] for i in best_indices]
        if self.search_all:
            best_results = self.remove_dup(best_results)
            if self.sort_results:
                best_results = sorted(best_results, key=lambda x: x.score, reverse=True)
        return best_results or [initial_result]

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["beam_width"]
