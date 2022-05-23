from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import BeamSearch
import numpy as np
import time


class BeamSearchWithTimeout(BeamSearch):
    """
    TextAttack's beam search but with timeout.
    If timeout is reached and we haven't completed searching all beams then stop search and return a failure for the attacked text

    An attack that maintinas a beam of the `beam_width` highest scoring
    AttackedTexts, greedily updating the beam with the highest scoring
    transformations from the current beam.

    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        transformation: The type of transformation.
        beam_width (int): the number of candidates to retain at each step
    """

    def __init__(self, beam_width=8, timeout=60, **kwargs):
        super().__init__(**kwargs)
        self.beam_width = beam_width
        self.timeout = timeout

    def perform_search(self, initial_result):
        beam = [initial_result.attacked_text]
        best_results = []
        search_over = False
        start = time.time()
        while not search_over:
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
            if self.search_all:
                best_results.extend(
                    list(filter(lambda x: x.goal_status == GoalFunctionResultStatus.SUCCEEDED, results)))
            else:
                scores = np.array([r.score for r in results])
                best_results = [results[scores.argmax()]]
                if best_results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    return best_results
            has_timed_out = (time.time() - start) > self.timeout
            if search_over or has_timed_out:
                return best_results

            # Refill the beam. This works by sorting the scores
            # in descending order and filling the beam from there.
            best_indices = (-scores).argsort()[: self.beam_width]
            beam = [potential_next_beam[i] for i in best_indices]

        if self.search_all and self.sort_results:
            best_results = sorted(best_results, key=lambda x: x.score, reverse=True)
        return best_results or [initial_result]
