class RecommendationConfig:
    """
    Configuration for the recommendation module.
    """

    def __init__(self, config, semantic_weight=0.5, relevance_thresh=0.5, action_thresh=0.8,
                 top_k=100):
        self.config = config
        self.semantic_weight = semantic_weight
        self.top_k = top_k
        self.relevance_thresh = relevance_thresh
        self.action_thresh = action_thresh
        self.generality_pattern = []
        self.relevance_pattern = []
        # self.generality_pattern.append(self.config.OBJECT_BASED_GENERALITY)
        # self.generality_pattern.append(self.config.NAME_BASED_GENERALITY)
        # self.generality_pattern.append(self.config.LABEL_BASED_GENERALITY)
        # #(x[[patt for patt in self.relevance_pattern]].max()) \ place this in function instead of SEMANTIC_BASED_RELEVANCE if needed
        self.relevance_pattern.append(self.config.SEMANTIC_BASED_RELEVANCE)

    def get_lambda_function(self, max_support):
        """
        Returns a lambda function that can be used to calculate the score of a constraint.
        :param constraints: The constraints that are used to calculate the score.
        :return: A lambda function that can be used to calculate the score of a constraint.

        Currently, the score is calculated as follows:

        """
        return lambda x: (1 - self.semantic_weight) * (
                x[self.config.SUPPORT] / max_support[x[self.config.LEVEL]]) + \
                         (self.semantic_weight * x[self.config.SEMANTIC_BASED_RELEVANCE] \
            if x[self.config.SUPPORT] > 0 and x[self.config.SEMANTIC_BASED_RELEVANCE] is not None else 0)

# (1 - self.frequency_weight) * (x[[patt for patt in self.generality_pattern]].max())
