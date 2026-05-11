"""Safe custom math channel evaluation and unit inference."""

import ast
import re

import numpy as np
import pandas as pd


def derivative(y, x=None):
    """Return dy/dx for a Series or array."""
    y_values = np.asarray(y, dtype=float)
    if x is None:
        x_values = np.arange(len(y_values), dtype=float)
    else:
        x_values = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.gradient(y_values, x_values)
    return pd.Series(result, index=getattr(y, "index", None)).replace([np.inf, -np.inf], np.nan)


def integral(y, x=None):
    """Return cumulative trapezoidal integral of y over x."""
    y_values = np.asarray(y, dtype=float)
    if x is None:
        x_values = np.arange(len(y_values), dtype=float)
    else:
        x_values = np.asarray(x, dtype=float)
    result = np.zeros(len(y_values), dtype=float)
    if len(y_values) > 1:
        dx = np.diff(x_values)
        avg_y = (y_values[1:] + y_values[:-1]) / 2.0
        result[1:] = np.cumsum(avg_y * dx)
    return pd.Series(result, index=getattr(y, "index", None))


class MathChannelEngine:
    """Evaluate calculator-style math expressions over telemetry channels."""

    CHANNEL_PATTERN = re.compile(r"\{([^{}]+)\}")
    FUNCTIONS = {
        "abs": np.abs,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
        "sqrt": np.sqrt,
        "log": np.log,
        "log10": np.log10,
        "exp": np.exp,
        "radians": np.radians,
        "degrees": np.degrees,
        "derivative": derivative,
        "integral": integral,
    }
    CONSTANTS = {
        "pi": np.pi,
        "e": np.e,
    }
    ALLOWED_NODES = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
    )

    @classmethod
    def evaluate(cls, expression, channel_resolver, index=None):
        python_expr, channel_names = cls._to_python_expression(expression)
        channel_values = {}
        for var_name, channel_name in channel_names.items():
            series = channel_resolver(channel_name)
            if series is None:
                raise KeyError(f"Could not resolve math channel input '{channel_name}'")
            channel_values[var_name] = series

        env = {**cls.FUNCTIONS, **cls.CONSTANTS, **channel_values}
        tree = cls._validated_tree(python_expr, env)
        result = eval(compile(tree, "<math-channel>", "eval"), {"__builtins__": {}}, env)

        if isinstance(result, pd.Series):
            return result
        if isinstance(result, np.ndarray):
            return pd.Series(result, index=index)
        return pd.Series(result, index=index)

    @classmethod
    def infer_unit(cls, expression, channel_units):
        python_expr, channel_names = cls._to_python_expression(expression)
        unit_env = {
            **{name: "" for name in cls.FUNCTIONS},
            **{name: "" for name in cls.CONSTANTS},
            **{var_name: channel_units.get(channel_name, "") for var_name, channel_name in channel_names.items()},
        }
        tree = cls._validated_tree(python_expr, unit_env)
        return cls._infer_node_unit(tree.body, unit_env)

    @classmethod
    def channel_names(cls, expression):
        return [match.group(1).strip() for match in cls.CHANNEL_PATTERN.finditer(expression)]

    @classmethod
    def _to_python_expression(cls, expression):
        channel_names = {}

        def replace(match):
            var_name = f"__ch{len(channel_names)}"
            channel_names[var_name] = match.group(1).strip()
            return var_name

        return cls.CHANNEL_PATTERN.sub(replace, expression), channel_names

    @classmethod
    def _validated_tree(cls, expression, env):
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, cls.ALLOWED_NODES):
                raise ValueError(f"Unsupported expression element: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id not in env:
                raise ValueError(f"Unknown name '{node.id}'. Use channel buttons or supported calculator functions.")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in cls.FUNCTIONS:
                    supported = ", ".join(sorted(cls.FUNCTIONS))
                    raise ValueError(f"Only supported calculator functions can be called: {supported}.")
        return tree

    @classmethod
    def _infer_node_unit(cls, node, unit_env):
        if isinstance(node, ast.Constant):
            return ""
        if isinstance(node, ast.Name):
            return unit_env.get(node.id, "")
        if isinstance(node, ast.UnaryOp):
            return cls._infer_node_unit(node.operand, unit_env)
        if isinstance(node, ast.BinOp):
            left = cls._infer_node_unit(node.left, unit_env)
            right = cls._infer_node_unit(node.right, unit_env)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return left if left == right else cls._join_mismatch(left, right)
            if isinstance(node.op, ast.Mult):
                return cls._multiply_units(left, right)
            if isinstance(node.op, ast.Div):
                return cls._divide_units(left, right)
            if isinstance(node.op, ast.Pow):
                return cls._power_unit(left, node.right)
            if isinstance(node.op, ast.Mod):
                return left
        if isinstance(node, ast.Call):
            name = node.func.id
            arg_unit = cls._infer_node_unit(node.args[0], unit_env) if node.args else ""
            second_arg_unit = cls._infer_node_unit(node.args[1], unit_env) if len(node.args) > 1 else ""
            if name in {"sin", "cos", "tan", "log", "log10", "exp"}:
                return ""
            if name in {"asin", "acos", "atan"}:
                return "rad"
            if name == "degrees":
                return "deg"
            if name == "radians":
                return "rad"
            if name == "sqrt":
                return f"sqrt({arg_unit})" if arg_unit else ""
            if name == "derivative":
                return cls._divide_units(arg_unit, second_arg_unit or "sample")
            if name == "integral":
                return cls._multiply_units(arg_unit, second_arg_unit or "sample")
            return arg_unit
        return ""

    @staticmethod
    def _join_mismatch(left, right):
        if not left:
            return right
        if not right:
            return left
        return f"{left} or {right}"

    @staticmethod
    def _multiply_units(left, right):
        if not left:
            return right
        if not right:
            return left
        return f"{left}*{right}"

    @staticmethod
    def _divide_units(left, right):
        if not right:
            return left
        if not left:
            return f"1/{right}"
        return f"{left}/{right}"

    @staticmethod
    def _power_unit(unit, exponent_node):
        if not unit:
            return ""
        if isinstance(exponent_node, ast.Constant):
            return f"{unit}^{exponent_node.value}"
        return f"{unit}^x"
