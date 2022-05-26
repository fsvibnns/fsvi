from functools import partial
from typing import Callable, List

import haiku as hk


def iterate_sub_params(
    fn_of_params: Callable, params: hk.Params, list_of_predicates: List[Callable]
):
    list_of_tuples = []
    for predicate in list_of_predicates:
        _sub_params, _rest_params = hk.data_structures.partition(predicate, params)

        def new_fn_of_params(sub_params, rest_params):
            full_params = hk.data_structures.merge(sub_params, rest_params)
            return fn_of_params(full_params)

        list_of_tuples.append((partial(new_fn_of_params, rest_params=_rest_params), _sub_params))
    return list_of_tuples


def predicates_by_layers(params: hk.Params) -> List[Callable]:
    keys = sorted(params.keys())

    def predicate(module_name, name, value, is_module_name):
        return module_name == is_module_name
    return [partial(predicate, is_module_name=n) for n in keys]
