from textattack.search_methods import GreedyWordSwapWIR
from textattack.goal_function_results import GoalFunctionResultStatus
import stopit


class GreedyWordSwapWIRWithTimeout(GreedyWordSwapWIR):
    """
    TextAttack's GreedyWordSwapWIR search but with timeout.
    If timeout is reached and we haven't completed searching then stop search and return a failure for the attacked text


    An attack that greedily chooses from a list of possible perturbations in
    order of index, after ranking indices by importance.

    Args:
        wir_method: method for ranking most important words
        model_wrapper: model wrapper used for gradient-based ranking
    """

    def __init__(self, wir_method="unk", timeout=90, **kwargs):
        super().__init__(**kwargs)
        self.wir_method = wir_method
        self.timeout = timeout

    def perform_search(self, initial_result):
        attacked_text = initial_result.attacked_text

        # Sort words by order of importance
        index_order, search_over = self._get_index_order(attacked_text)

        i = 0
        cur_result = initial_result
        best_results = []
        with stopit.ThreadingTimeout(self.timeout) as to_ctx_mgr:
            assert to_ctx_mgr.state == to_ctx_mgr.EXECUTING
            while i < len(index_order) and not search_over:
                transformed_text_candidates = self.get_transformations(
                    cur_result.attacked_text,
                    original_text=initial_result.attacked_text,
                    indices_to_modify=[index_order[i]],
                )
                i += 1
                if len(transformed_text_candidates) == 0:
                    continue
                results, search_over = self.get_goal_results(
                    transformed_text_candidates)
                results = sorted(results, key=lambda x: -x.score)
                # Skip swaps which don't improve the score
                if results[0].score > cur_result.score:
                    cur_result = results[0]
                else:
                    continue

                if self.search_all:
                    best_results.extend(
                        list(filter(lambda x: x.goal_status == GoalFunctionResultStatus.SUCCEEDED, results)))
                    continue

                # If we succeeded, return the index with best similarity.
                if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    best_result = cur_result
                    # @TODO: Use vectorwise operations
                    max_similarity = -float("inf")
                    for result in results:
                        if result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
                            break
                        candidate = result.attacked_text
                        try:
                            similarity_score = candidate.attack_attrs["similarity_score"]
                        except KeyError:
                            # If the attack was run without any similarity metrics,
                            # candidates won't have a similarity score. In this
                            # case, break and return the candidate that changed
                            # the original score the most.
                            break
                        if similarity_score > max_similarity:
                            max_similarity = similarity_score
                            best_result = result
                    return [best_result]
        best_results = self.remove_dup(best_results) or [initial_result]
        if self.search_all and self.sort_results:
            best_results = sorted(best_results, key=lambda x: x.score, reverse=True)
        return best_results if self.search_all else [cur_result]
