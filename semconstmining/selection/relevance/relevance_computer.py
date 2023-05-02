import logging
from statistics import mean

import pandas as pd
from pandas import DataFrame

from semconstmining.parsing.label_parser import nlp_helper
from semconstmining.parsing.resource_handler import write_pickle

_logger = logging.getLogger(__name__)


class RelevanceComputer:

    def __init__(self, config, nlp_helper, resource_handler, log_info):
        self.config = config
        self.nlp_helper = nlp_helper
        self.resource_handler = resource_handler
        self.log_info = log_info

    def compute_relevance(self, constraints):
        _logger.info("Computing relevance")
        constraints = constraints.copy(deep=True)
        self.nlp_helper.pre_compute_embeddings(constraints, self.resource_handler,
                                               sentences=self.log_info.labels + self.log_info.names +
                                                         list(self.log_info.resources_to_tasks.keys()) + self.log_info.objects +
                                                         self.log_info.actions)
        constraints[self.config.INDIVIDUAL_RELEVANCE_SCORES] = \
            constraints.apply(lambda row: self._compute_relevance(row), axis=1)
        constraints[self.config.SEMANTIC_BASED_RELEVANCE] = constraints.apply(lambda row: self.get_max_scores(row),
                                                                              axis=1)
        return constraints

    def _compute_relevance(self, row):
        if row[self.config.LEVEL] == self.config.OBJECT:
            return self.get_relevance_for_object_constraint(row)
        elif row[self.config.LEVEL] == self.config.MULTI_OBJECT:
            return self.get_relevance_for_multi_object_constraint(row)
        elif row[self.config.LEVEL] == self.config.ACTIVITY:
            return self.get_relevance_for_activity_constraint(row)
        elif row[self.config.LEVEL] == self.config.RESOURCE:
            return self.get_relevance_for_resource_constraint(row)

    def get_relevance_for_object_constraint(self, row):
        object_sims = {self.config.OBJECT: {}, self.config.ACTION: {}}
        for ext in self.log_info.objects:
            combi = [(row[self.config.OBJECT], ext)]
            object_sims[self.config.OBJECT][ext] = self.nlp_helper.get_sims(combi)[0]
        for ext in self.log_info.actions:
            synonyms = self.nlp_helper.get_synonyms(ext)
            if not pd.isna(row[self.config.LEFT_OPERAND]) and not row[
                                                                      self.config.LEFT_OPERAND] in self.config.TERMS_FOR_MISSING:
                if row[self.config.LEFT_OPERAND] in synonyms:
                    object_sims[self.config.ACTION][row[self.config.LEFT_OPERAND]] = ext
            if not pd.isna(row[self.config.RIGHT_OPERAND]) and not row[
                                                                       self.config.RIGHT_OPERAND] in self.config.TERMS_FOR_MISSING:
                if row[self.config.RIGHT_OPERAND] in synonyms:
                    object_sims[self.config.ACTION][row[self.config.RIGHT_OPERAND]] = ext
        return object_sims

    def get_relevance_for_multi_object_constraint(self, row):
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

    def get_relevance_for_activity_constraint(self, row):
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

    def get_relevance_for_resource_constraint(self, row):
        label_sims = {self.config.LEFT_OPERAND: {}, self.config.RESOURCE: {}}
        if not pd.isna(row[self.config.LEFT_OPERAND]) and not row[
                                                                  self.config.LEFT_OPERAND] in self.config.TERMS_FOR_MISSING:
            label_sims[self.config.LEFT_OPERAND] = {}
        for ext in self.log_info.labels:
            if self.config.LEFT_OPERAND in label_sims:
                combi = [(row[self.config.LEFT_OPERAND], ext)]
                label_sims[self.config.LEFT_OPERAND][ext] = self.nlp_helper.get_sims(combi)[0]
        for ext in self.log_info.resources_to_tasks:
            combi = [(row[self.config.RESOURCE], ext)]
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