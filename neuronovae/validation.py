import inspect
from functools import wraps
from typing import Any, Callable

__all__ = [
    "convert",
]


"""
This module contains various tools that can be used to validate user input
"""


def parameterize(decorator: Callable) -> Callable:
    """
    Function for parameterizing decorators

    :param decorator: A decorator to parameterize

    :returns: A decorator that can be parameterized
    """

    def outer(*args, **kwargs) -> Callable:
        """
        Outer function that takes arguments and keyword arguments for the decorator

        :param args: Positional arguments for the decorator

        :param kwargs: Keyword arguments for the decorator

        :returns: A function that applies the decorator to the target function
        """

        def inner(func: Callable) -> Callable:
            """
            Inner function that applies the decorator to the target function

            :param func: The target function to be decorated

            :returns: The decorated function
            """
            # noinspection PyArgumentList
            return decorator(func, *args, **kwargs)

        return inner

    return outer


@parameterize
def convert(
    function: Callable,
    parameter: str,
    permitted: tuple[Any, ...],
    required: Any,
    converter: Callable | None = None,
) -> Callable:
    """
    Decorator for converting a parameter to a required type. Useful when coercion is not
    straightforward.

    :param function: The function to be decorated.
    :param parameter: The parameter to be converted.
    :param permitted: The types from which to convert to the desired type
    :param required: The type to which the parameter should be converted
    :param converter: A function to convert the parameter to the required type
    :returns: The decorated function
    """

    @wraps(function)
    def decorator(*args, **kwargs) -> Callable:
        """
        Inner decorator function that performs the conversion. We make sure to consider
        both positional and keyword arguments.
        """
        sig = inspect.signature(function)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        param = bound_args.arguments.get(parameter)
        if isinstance(param, permitted):
            bound_args.arguments = {**bound_args.kwargs, **bound_args.arguments}
            bound_args.arguments.pop("kwargs", None)
            bound_args.arguments[parameter] = (
                converter(param) if converter else required(param)
            )
        else:
            raise TypeError(f"{parameter} must be of type {permitted}")
        return function(**bound_args.arguments)

    return decorator
