import logging
import uuid

import pandas as pd
from pm4py.objects.log.obj import EventLog, Trace, Event

from semconstmining.constraintmining.conversion.petrinetanalysis import _is_relevant_label
from semconstmining.constraintmining.model.parsed_label import get_dummy
from semconstmining.declare.declare import Declare4Py
from semconstmining.parsing.resource_handler import ResourceHandler

_logger = logging.getLogger(__name__)


class DeclareExtractor:

    def __init__(self, config, resource_handler: ResourceHandler, types_to_ignore=None):
        self.config = config
        self.resource_handler = resource_handler
        self.types_to_ignore = [] if types_to_ignore is None else types_to_ignore
        _logger.info("Will ignore: " + str(self.types_to_ignore))

    def add_operands(self, res):
        for rec in res:
            if rec[self.config.OPERATOR_TYPE] == self.config.UNARY:
                op_l = rec[self.config.config.CONSTRAINT_STR].split("[")[1].replace("]", "").replace("|", "").strip()
                rec[self.config.config.LEFT_OPERAND] = op_l if op_l not in self.config.TERMS_FOR_MISSING else ""
            if rec[self.config.OPERATOR_TYPE] == self.config.BINARY:
                ops = rec[self.config.CONSTRAINT_STR].split("[")[1].replace("]", "").replace("|", "").split(",")
                op_l = ops[0].strip()
                op_r = ops[1].strip()
                rec[self.config.LEFT_OPERAND] = op_l if op_l not in self.self.config.TERMS_FOR_MISSING else ""
                rec[self.config.RIGHT_OPERAND] = op_r if op_r not in self.config.TERMS_FOR_MISSING else ""
        return res

    def get_object_constraints_flat(self, res):
        res = [{self.config.RECORD_ID: str(uuid.uuid4()),
                self.config.LEVEL: self.config.OBJECT,
                self.config.OBJECT: bo,
                self.config.CONSTRAINT_STR: const,
                self.config.OPERATOR_TYPE: self.config.BINARY if any(temp in const for temp in self.config.BINARY_TEMPLATES) else self.config.UNARY
                } for bo, consts in res.items() for const in consts]
        res = self.add_operands(res)
        for rec in res:
            if rec[self.config.OPERATOR_TYPE] == self.config.UNARY:
                rec[self.config.LEFT_OPERAND] = rec[self.config.CONSTRAINT_STR].split("[")[1].replace("]", "").replace("|", "").strip()
            if rec[self.config.OPERATOR_TYPE] == self.config.BINARY:
                ops = rec[self.config.CONSTRAINT_STR].split("[")[1].replace("]", "").replace("|", "").split(",")
                rec[self.config.LEFT_OPERAND] = ops[0].strip()
                rec[self.config.RIGHT_OPERAND] = ops[1].strip()
        return res

    def get_multi_object_constraints_flat(self, res):
        res = [{self.config.RECORD_ID: str(uuid.uuid4()),
                self.config.LEVEL: self.config.MULTI_OBJECT,
                self.config.OBJECT: "",
                self.config.CONSTRAINT_STR: const,
                self.config.OPERATOR_TYPE: self.config.BINARY if any(temp in const for temp in self.config.BINARY_TEMPLATES) else self.config.UNARY
                } for const in res]
        for rec in res:
            if rec[self.config.OPERATOR_TYPE] == self.config.UNARY:
                rec[self.config.LEFT_OPERAND] = rec[self.config.CONSTRAINT_STR].split("[")[1].replace("]", "").replace("|", "").strip()
            if rec[self.config.OPERATOR_TYPE] == self.config.BINARY:
                ops = rec[self.config.CONSTRAINT_STR].split("[")[1].replace("]", "").replace("|", "").split(",")
                rec[self.config.LEFT_OPERAND] = ops[0].strip()
                rec[self.config.RIGHT_OPERAND] = ops[1].strip()
        return res

    def extract_declare_from_logs(self):
        """
        Extract DECLARE-like constraints from log traces
        :return: a pandas dataframe with extracted DECLARE constraints
        """
        _logger.info("Extracting DECLARE constraints from played-out logs")
        # Discover action based constraints per object
        dfs_obj = [self.discover_object_based_declare_constraints(t) for t in
                   self.resource_handler.bpmn_logs.reset_index().itertuples()]
        # Discover multi-object constraints
        dfs_multi_obj = [self.discover_multi_object_declare_constraints(t) for t in
                         self.resource_handler.bpmn_logs.reset_index().itertuples()]

        # Combine all constraints that were extracted into a common dataframe
        dfs = [df for df in dfs_obj + dfs_multi_obj if df is not None]
        new_df = pd.concat(dfs).astype({self.config.LEVEL: "category"})
        return new_df

    def discover_multi_object_declare_constraints(self, row_tuple):
        if row_tuple.log is None:
            return None
        parsed_tasks = self.get_parsed_tasks(row_tuple.log, resource_handler=self.resource_handler)
        filtered_traces = self.get_filtered_traces(row_tuple.log, parsed_tasks=parsed_tasks)
        res = set()
        d4py = Declare4Py()
        d4py.log = self.object_log_projection(filtered_traces)
        d4py.compute_frequent_itemsets(min_support=0.99, len_itemset=2)
        individual_res = d4py.discovery(consider_vacuity=False, max_declare_cardinality=2)
        res.update(const for const, checker_results in individual_res.items()
                   if "[]" not in const and "[none]" not in const
                   and ''.join([i for i in const.split("[")[0] if not i.isdigit()]) not in self.types_to_ignore)
        return (
            pd.DataFrame.from_records(self.get_multi_object_constraints_flat(res)).assign(
                model_id=row_tuple.model_id).assign(
                model_name=row_tuple.name)
        )

    def discover_object_based_declare_constraints(self, row_tuple):
        if row_tuple.log is None:
            return None
        parsed_tasks = self.get_parsed_tasks(row_tuple.log, resource_handler=self.resource_handler)
        filtered_traces = self.get_filtered_traces(row_tuple.log, parsed_tasks=parsed_tasks)
        res = {}
        bos = set([x.main_object for trace in filtered_traces for x in trace if x.main_object not in self.config.TERMS_FOR_MISSING])
        # _logger.info(bos)
        for bo in bos:
            d4py = Declare4Py()
            d4py.log = self.object_action_log_projection(bo, filtered_traces)
            d4py.compute_frequent_itemsets(min_support=0.99, len_itemset=2)
            individual_res = d4py.discovery(consider_vacuity=False, max_declare_cardinality=2)
            # print(individual_res)
            if bo not in res:
                res[bo] = set()
            res[bo].update(const for const, checker_results in individual_res.items() if "[]" not in const
                           and "[none]" not in const
                           and ''.join([i for i in const.split("[")[0] if not i.isdigit()]) not in self.types_to_ignore)
        return (
            pd.DataFrame.from_records(self.get_object_constraints_flat(res)).assign(model_id=row_tuple.model_id).assign(
                model_name=row_tuple.name)
        )

    def discover_declare_constraints(self, row_tuple):
        if row_tuple.log is None:
            return None
        d4py = Declare4Py()
        d4py.log = row_tuple.log  # get_filtered_traces(row_tuple.log, with_loops=True)
        d4py.compute_frequent_itemsets(min_support=0.99, len_itemset=2)
        res = d4py.discovery(consider_vacuity=False, max_declare_cardinality=2)
        res = {const for const, checker_results in res.items() if "[]" not in const
               and "[none]" not in const
               and ''.join([i for i in const.split("[")[0] if not i.isdigit()]) not in self.types_to_ignore}
        return res

    def has_loop(self, trace):
        trace_labels = [x[self.config.XES_NAME] for x in trace]
        return len(trace_labels) > len(set(trace_labels))

    def get_parsed_tasks(self, log, resource_handler, only_relevant_labels=True):
        relevant_tasks = set([x[self.config.XES_NAME] for trace in log for x in trace if
                              _is_relevant_label(x[self.config.XES_NAME])]) if only_relevant_labels else set(
            [x[self.config.XES_NAME] for trace in log for x in trace])
        return {t: resource_handler.get_parsed_task(t) for t in relevant_tasks}

    def get_filtered_traces(self, log, parsed_tasks=None, with_loops=False):
        if parsed_tasks is not None:
            return [
                [parsed_tasks[e[self.config.XES_NAME]] if e[self.config.XES_NAME] in parsed_tasks else get_dummy(self.config, e[self.config.XES_NAME], self.config.EN) for i, e in
                 enumerate(trace)] for trace in log if not with_loops and not self.has_loop(trace)]
        else:
            return [[e[self.config.XES_NAME] for i, e in enumerate(trace)] for trace in log if
                    not with_loops and not self.has_loop(trace)]

    def object_action_log_projection(self, obj, traces):
        """
        Return for each trace a time-ordered list of the actions for a given object type.

        Returns
        -------
        projection
            traces containing only actions applied to the same obj.
        """
        projection = EventLog()
        if traces is None:
            raise RuntimeError("You must load a log before.")
        for i, trace in enumerate(traces):
            tmp_trace = Trace()
            tmp_trace.attributes[self.config.XES_NAME] = str(i)
            for parsed in trace:
                if parsed.main_object == obj:
                    if parsed.main_action != "":
                        event = Event({self.config.XES_NAME: parsed.main_action})
                        tmp_trace.append(event)
            if len(tmp_trace) > 0:
                projection.append(tmp_trace)
        return projection

    def object_log_projection(self, traces):
        """
        Return for each trace a time-ordered list of the actions for a given object type.

        Returns
        -------
        projection
            traces containing only actions applied to the same obj.
        """
        projection = EventLog()
        if traces is None:
            raise RuntimeError("You must load a log before.")
        for i, trace in enumerate(traces):
            tmp_trace = Trace()
            tmp_trace.attributes[self.config.XES_NAME] = str(i)
            last = ""
            for parsed in trace:
                if type(parsed.main_object) != str:
                    _logger.warning("main_object is not a string: %s" % parsed.main_object)
                    continue
                if parsed.main_object not in self.config.TERMS_FOR_MISSING and parsed.main_object != last:
                    event = Event({self.config.XES_NAME: parsed.main_object})
                    tmp_trace.append(event)
                last = parsed.main_object
            projection.append(tmp_trace)
        return projection
