import itertools
import logging
import pickle
from os.path import exists

import pandas as pd

_logger = logging.getLogger(__name__)


class RelevanceComputer:

    def __init__(self, config, nlp_helper, log_info):
        self.config = config
        self.nlp_helper = nlp_helper
        self.sims = {}
        self.log_info = log_info
        self.counter = 0

    def compute_relevance(self, constraints, pre_compute=False, store_sims=False):
        _logger.info(f"Computing relevance for {len(constraints)} constraints")
        constraints = constraints.copy(deep=True)
        file_name = (self.config.MODEL_COLLECTION + "_relevance_" + str(self.log_info.log_id) + ".pkl")
        if exists(self.config.DATA_INTERIM / file_name):
            _logger.info(f"Loading relevance from {file_name}")
            constraints[self.config.INDIVIDUAL_RELEVANCE_SCORES] = pickle.load(
                open(self.config.DATA_INTERIM / file_name, "rb"))
        else:
            if pre_compute:
                self.nlp_helper.pre_compute_embeddings(sentences=self.log_info.labels + self.log_info.names +
                                                                 list(self.log_info.resources_to_tasks.keys()) +
                                                                 self.log_info.objects + self.log_info.actions)
            self.sims = self.precompute_sims(constraints)
            levels_leftop_rightop_obj = zip(
                constraints[self.config.LEVEL].tolist(),
                constraints[self.config.LEFT_OPERAND].tolist(),
                constraints[self.config.RIGHT_OPERAND].tolist(),
                constraints[self.config.OBJECT].tolist()
            )
            constraints[self.config.INDIVIDUAL_RELEVANCE_SCORES] = pd.Series(
                [self._compute_relevance(level, left_op, right_op, obj) for (index, (level, left_op, right_op, obj))
                        in enumerate(levels_leftop_rightop_obj)],
                index= constraints.index)

#           constraints[self.config.INDIVIDUAL_RELEVANCE_SCORES] = \
#               constraints.apply(lambda row: self._compute_relevance_row(row), axis=1)
            if store_sims:
                pickle.dump(constraints[self.config.INDIVIDUAL_RELEVANCE_SCORES],
                            open(self.config.DATA_INTERIM / file_name, "wb"))

        constraints = constraints[~constraints[self.config.INDIVIDUAL_RELEVANCE_SCORES].isna()]
        constraints[self.config.SEMANTIC_BASED_RELEVANCE] = constraints.apply(lambda row: self.get_max_scores(row),
                                                                              axis=1)
        # self.nlp_helper.store_sims()
        return constraints
    
    def _compute_relevance(self, level, left_op, right_op, obj):
        self.counter += 1
        if self.counter % 500 == 0:
            _logger.info(f"Computing relevance for constraint number {self.counter}")
        if level == self.config.OBJECT:
            return self.get_relevance_for_object_constraint(obj, left_op, right_op)
        elif level == self.config.MULTI_OBJECT:
            return self.get_relevance_for_multi_object_constraint(left_op, right_op)
        elif level == self.config.ACTIVITY:
            return self.get_relevance_for_activity_constraint(left_op, right_op)
        elif level == self.config.RESOURCE:
            return self.get_relevance_for_resource_constraint(left_op, obj)

    def _compute_relevance_row(self, row):
        self.counter += 1
        if self.counter % 500 == 0:
            _logger.info(f"Computing relevance for constraint number {self.counter}")
        if row[self.config.LEVEL] == self.config.OBJECT:
            return self.get_relevance_for_object_constraint_row(row)
        elif row[self.config.LEVEL] == self.config.MULTI_OBJECT:
            return self.get_relevance_for_multi_object_constraint_row(row)
        elif row[self.config.LEVEL] == self.config.ACTIVITY:
            return self.get_relevance_for_activity_constraint_row(row)
        elif row[self.config.LEVEL] == self.config.RESOURCE:
            return self.get_relevance_for_resource_constraint_row(row)
        
    def get_relevance_for_object_constraint(self, obj, left_op, right_op):
        object_sims = {self.config.OBJECT: {}, self.config.ACTION: {}}
        for ext in self.log_info.objects:
            combi = [(obj, ext)]
            object_sims[self.config.OBJECT][ext] = self.nlp_helper.get_sims(combi)[0]
        for ext in self.log_info.actions:
            synonyms = self.nlp_helper.get_synonyms(ext)
            similar_actions = self.nlp_helper.get_similar_actions(ext)
            if left_op is not None and left_op not in self.config.TERMS_FOR_MISSING:
                if left_op in synonyms or left_op in similar_actions:
                    object_sims[self.config.ACTION][left_op] = ext
            if right_op is not None and right_op not in self.config.TERMS_FOR_MISSING:
                if right_op in synonyms or right_op in similar_actions:
                    object_sims[self.config.ACTION][right_op] = ext
        return object_sims

    def get_relevance_for_object_constraint_row(self, row):
        object_sims = {self.config.OBJECT: {}, self.config.ACTION: {}}
        for ext in self.log_info.objects:
            combi = [(row[self.config.OBJECT], ext)]
            object_sims[self.config.OBJECT][ext] = self.nlp_helper.get_sims(combi)[0]
        for ext in self.log_info.actions:
            synonyms = self.nlp_helper.get_synonyms(ext)
            similar_actions = self.nlp_helper.get_similar_actions(ext)
            if not pd.isna(row[self.config.LEFT_OPERAND]) and not row[
                                                                      self.config.LEFT_OPERAND] in self.config.TERMS_FOR_MISSING:
                if row[self.config.LEFT_OPERAND] in synonyms or row[self.config.LEFT_OPERAND] in similar_actions:
                    object_sims[self.config.ACTION][row[self.config.LEFT_OPERAND]] = ext
            if not pd.isna(row[self.config.RIGHT_OPERAND]) and not row[
                                                                       self.config.RIGHT_OPERAND] in self.config.TERMS_FOR_MISSING:
                if row[self.config.RIGHT_OPERAND] in synonyms or row[self.config.RIGHT_OPERAND] in similar_actions:
                    object_sims[self.config.ACTION][row[self.config.RIGHT_OPERAND]] = ext
        return object_sims
    
    def get_relevance_for_multi_object_constraint(self, left_op, right_op):
        object_sims = {}
        if left_op is not None and not left_op in self.config.TERMS_FOR_MISSING:
            object_sims[self.config.LEFT_OPERAND] = {}
        if right_op is not None and not right_op in self.config.TERMS_FOR_MISSING:
            object_sims[self.config.RIGHT_OPERAND] = {}

        for ext in self.log_info.objects:
            if self.config.LEFT_OPERAND in object_sims:
                combi = [(left_op, ext)]
                object_sims[self.config.LEFT_OPERAND][ext] = self.nlp_helper.get_sims(combi)[0]
            if self.config.RIGHT_OPERAND in object_sims:
                combi = [(right_op, ext)]
                object_sims[self.config.RIGHT_OPERAND][ext] = self.nlp_helper.get_sims(combi)[0]
        return object_sims

    def get_relevance_for_multi_object_constraint_row(self, row):
        object_sims = {}
        if not pd.isna(row[self.config.LEFT_OPERAND]) and not row[
                                                                  self.config.LEFT_OPERAND] in self.config.TERMS_FOR_MISSING:
            object_sims[self.config.LEFT_OPERAND] = {}
        if not pd.isna(row[self.config.RIGHT_OPERAND]) and not row[
                                                                   self.config.RIGHT_OPERAND] in self.config.TERMS_FOR_MISSING:
            object_sims[self.config.RIGHT_OPERAND] = {}
        for ext in self.log_info.objects:
            if self.config.LEFT_OPERAND in object_sims:
                combi = [(row[self.config.LEFT_OPERAND], ext)]
                object_sims[self.config.LEFT_OPERAND][ext] = self.nlp_helper.get_sims(combi)[0]
            if self.config.RIGHT_OPERAND in object_sims:
                combi = [(row[self.config.RIGHT_OPERAND], ext)]
                object_sims[self.config.RIGHT_OPERAND][ext] = self.nlp_helper.get_sims(combi)[0]
        return object_sims

    def get_relevance_for_activity_constraint(self, left_op , right_op):
        label_sims = {}
        if left_op is not None and not left_op in self.config.TERMS_FOR_MISSING:
            label_sims[self.config.LEFT_OPERAND] = {}
        if right_op is not None and not right_op in self.config.TERMS_FOR_MISSING:
            label_sims[self.config.RIGHT_OPERAND] = {}
        for ext in self.log_info.labels:
            if self.config.LEFT_OPERAND in label_sims:
                combi = [(left_op, ext)]
                label_sims[self.config.LEFT_OPERAND][ext] = self.nlp_helper.get_sims(combi)[0]
            if self.config.RIGHT_OPERAND in label_sims:
                combi = [(right_op, ext)]
                label_sims[self.config.RIGHT_OPERAND][ext] = self.nlp_helper.get_sims(combi)[0]
        return label_sims
    
    def get_relevance_for_activity_constraint_row(self, row):
        label_sims = {}
        if not pd.isna(row[self.config.LEFT_OPERAND]) and not row[
                                                                  self.config.LEFT_OPERAND] in self.config.TERMS_FOR_MISSING:
            label_sims[self.config.LEFT_OPERAND] = {}
        if not pd.isna(row[self.config.RIGHT_OPERAND]) and not row[
                                                                   self.config.RIGHT_OPERAND] in self.config.TERMS_FOR_MISSING:
            label_sims[self.config.RIGHT_OPERAND] = {}
        for ext in self.log_info.labels:
            if self.config.LEFT_OPERAND in label_sims:
                combi = [(row[self.config.LEFT_OPERAND], ext)]
                label_sims[self.config.LEFT_OPERAND][ext] = self.nlp_helper.get_sims(combi)[0]
            if self.config.RIGHT_OPERAND in label_sims:
                combi = [(row[self.config.RIGHT_OPERAND], ext)]
                label_sims[self.config.RIGHT_OPERAND][ext] = self.nlp_helper.get_sims(combi)[0]
        return label_sims
    
    def get_relevance_for_resource_constraint(self, left_op, obj):
        label_sims = {self.config.LEFT_OPERAND: {}, self.config.RESOURCE: {}}
        if left_op is not None and not left_op in self.config.TERMS_FOR_MISSING:
            label_sims[self.config.LEFT_OPERAND] = {}
        for ext in self.log_info.labels:
            if self.config.LEFT_OPERAND in label_sims:
                combi = [(left_op, ext)]
                label_sims[self.config.LEFT_OPERAND][ext] = self.nlp_helper.get_sims(combi)[0]
        for ext in self.log_info.resources_to_tasks:
            combi = [(obj, ext)]
            label_sims[self.config.RESOURCE][ext] = self.nlp_helper.get_sims(combi)[0]
        return label_sims

    def get_relevance_for_resource_constraint_row(self, row):
        label_sims = {self.config.LEFT_OPERAND: {}, self.config.RESOURCE: {}}
        if not pd.isna(row[self.config.LEFT_OPERAND]) and not row[
                                                                  self.config.LEFT_OPERAND] in self.config.TERMS_FOR_MISSING:
            label_sims[self.config.LEFT_OPERAND] = {}
        for ext in self.log_info.labels:
            if self.config.LEFT_OPERAND in label_sims:
                combi = [(row[self.config.LEFT_OPERAND], ext)]
                label_sims[self.config.LEFT_OPERAND][ext] = self.nlp_helper.get_sims(combi)[0]
        for ext in self.log_info.resources_to_tasks:
            combi = [(row[self.config.OBJECT], ext)]
            label_sims[self.config.RESOURCE][ext] = self.nlp_helper.get_sims(combi)[0]
        return label_sims

    def get_max_scores(self, row):
        score = 0.0
        sim_dict = row[self.config.INDIVIDUAL_RELEVANCE_SCORES]
        if self.config.OBJECT in sim_dict and len(sim_dict[self.config.OBJECT]) > 0:
            score = max(sim_dict[self.config.OBJECT].values())
        if self.config.LEFT_OPERAND in sim_dict and len(sim_dict[self.config.LEFT_OPERAND]) > 0:
            score = max(sim_dict[self.config.LEFT_OPERAND].values())
        if self.config.RIGHT_OPERAND in sim_dict and len(sim_dict[self.config.RIGHT_OPERAND]) > 0:
            new_score = max(sim_dict[self.config.RIGHT_OPERAND].values())
            if new_score > score:
                score = new_score
        return score

    def precompute_sims(self, constraints):
        objects = list(
            constraints[constraints[self.config.LEVEL] == self.config.OBJECT][self.config.OBJECT].dropna().unique())
        objects += list(constraints[constraints[self.config.LEVEL] == self.config.MULTI_OBJECT][
                            self.config.LEFT_OPERAND].dropna().unique())
        objects += list(constraints[constraints[self.config.LEVEL] == self.config.MULTI_OBJECT][
                            self.config.RIGHT_OPERAND].dropna().unique())
        objects = list(set(objects))
        labels = list(constraints[constraints[self.config.LEVEL] == self.config.ACTIVITY][
                          self.config.LEFT_OPERAND].dropna().unique())
        labels += list(constraints[constraints[self.config.LEVEL] == self.config.ACTIVITY][
                           self.config.RIGHT_OPERAND].dropna().unique())
        labels += list(constraints[constraints[self.config.LEVEL] == self.config.RESOURCE][
                           self.config.LEFT_OPERAND].dropna().unique())
        labels = list(set(labels))
        resources = list(
            constraints[constraints[self.config.LEVEL] == self.config.RESOURCE][self.config.OBJECT].dropna().unique())
        object_combis = [(x, y) for x, y in itertools.product(objects, self.log_info.objects) if x != y]
        label_combis = [(x, y) for x, y in itertools.product(labels, self.log_info.labels) if x != y]
        resource_combis = [(x, y) for x, y in itertools.product(resources, self.log_info.resources_to_tasks) if x != y]
        _logger.info(
            "Precomputing similarities for {} object combinations, {} label combinations and {} resource combinations".format(
                len(object_combis), len(label_combis), len(resource_combis)))
        sims = self.nlp_helper.get_sims(object_combis)
        sims += self.nlp_helper.get_sims(label_combis)
        sims += self.nlp_helper.get_sims(resource_combis)
        return {x: y for x, y in zip(object_combis + label_combis + resource_combis, sims)}
