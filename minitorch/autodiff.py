from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # Приближаем по i-му элементу, соответственно, нам надо сначала
    # изменить i-й элемент
    # Приведем к листу (в тайп хинте указан любой обьект, видимо, итерируемый):
    vals = [i for i in vals]
    f_x = f(*vals)
    vals[arg] += epsilon
    f_x_p_eps = f(*vals)
    return (f_x_p_eps - f_x) / epsilon


variable_count = 1


class Variable(Protocol):
    def __init__(self, back):
        global variable_count
        super().__init__()
        self.id = variable_count
        variable_count += 1
        self.history = back
        self.derivative = None
        
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # Реализуем стандвртный алгоритм:

    res = []
    visited = set()

    def go_recurse(vertex):
        if vertex.unique_id in visited:
            return
        visited.add(vertex.unique_id)
        for parent in vertex.parents:
            go_recurse(parent)
        res.append(vertex)
    go_recurse(variable)
    return res[::-1]
            


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    topsort = topological_sort(variable)
    data = dict()
    data[variable.unique_id] = deriv
    for cur in topsort:
        x = data[cur.unique_id]
        if not cur.is_leaf():
            for s, part_x in cur.chain_rule(x):
                if s.unique_id not in data:
                    data[s.unique_id] = 0
                data[s.unique_id] += part_x
        else:
            cur.accumulate_derivative(x)

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
