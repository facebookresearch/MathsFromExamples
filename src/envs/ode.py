# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import io
import sys
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import sympy as sp
from sympy.core.cache import clear_cache
import control as ctrl
from scipy.linalg import expm
from scipy.integrate import cumtrapz
import scipy.optimize as opt

from ..utils import bool_flag
from ..utils import timeout, TimeoutError


CLEAR_SYMPY_CACHE_FREQ = 10000

SPECIAL_WORDS = ["<s>", "</s>", "<pad>", "(", ")"]
SPECIAL_WORDS = SPECIAL_WORDS + [f"<SPECIAL_{i}>" for i in range(10)]


logger = getLogger()


class UnknownSymPyOperator(Exception):
    pass


class InvalidPrefixExpression(Exception):
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return repr(self.data)


def has_inf_nan(*args):
    """
    Detect whether some SymPy expressions contain a NaN / Infinity symbol.
    """
    for f in args:
        if f.has(sp.nan) or f.has(sp.oo) or f.has(-sp.oo) or f.has(sp.zoo):
            return True
    return False


def second_index(x, bal):
    if bal not in x:
        return len(x)
    p1 = x.index(bal)
    if bal not in x[p1 + 1 :]:
        return len(x)
    p2 = x[p1 + 1 :].index(bal)
    return p2 + p1


def simplify(f, seconds):
    """
    Simplify an expression.
    """
    assert seconds > 0

    @timeout(seconds)
    def _simplify(f):
        try:
            f2 = sp.simplify(f)
            if any(s.is_Dummy for s in f2.free_symbols):
                logger.warning(f"Detected Dummy symbol when simplifying {f} to {f2}")
                return f
            else:
                return f2
        except TimeoutError:
            return f
        except Exception as e:
            logger.warning(f"{type(e).__name__} exception when simplifying {f}")
            return f

    return _simplify(f)


def expr_to_fun_real(x, fun, dimension):
    # for i in range(dimension):
    #     v='x'+str(i+1)
    #     v=sp.symbols('x'+str(i))
    #     f=f.subs(v,x[i])
    Eval = OrderedDict({sp.Symbol(f"x{i}"): x[i] for i in range(dimension)})
    fun = sp.re(fun.subs(Eval)).evalf()
    fun = min(fun, 1e15)
    fun = max(fun, -1e15)
    return fun


class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children else []

    def push_child(self, child):
        self.children.append(child)

    def prefix(self):
        s = str(self.value)
        for c in self.children:
            s += ", " + c.prefix()
        return s

    # export to latex qtree format: prefix with \Tree, use package qtree
    def qtree_prefix(self):
        s = "[.$" + str(self.value) + "$ "
        for c in self.children:
            s += c.qtree_prefix()
        s += "]"
        return s

    def infix(self):
        nb_children = len(self.children)
        if nb_children <= 1:
            s = str(self.value)
            if nb_children == 1:
                s += "(" + self.children[0].infix() + ")"
            return s
        s = "(" + self.children[0].infix()
        for c in self.children[1:]:
            s = s + " " + str(self.value) + " " + c.infix()
        return s + ")"

    def __len__(self):
        lenc = 1
        for c in self.children:
            lenc += len(c)
        return lenc

    def __str__(self):
        # infix a default print
        return self.infix()


class ODEEnvironment(object):

    TRAINING_TASKS = {
        "ode_convergence_speed",
        "ode_control",
        "fourier_cond_init",
    }

    def __init__(self, params):

        self.max_degree = params.max_degree
        self.min_degree = params.min_degree
        assert self.min_degree >= 2
        assert self.max_degree >= self.min_degree

        self.max_ops = 200

        self.max_int = params.max_int
        self.positive = params.positive
        self.nonnull = params.nonnull
        self.predict_jacobian = params.predict_jacobian
        self.predict_gramian = params.predict_gramian
        self.qualitative = params.qualitative
        self.allow_complex = params.allow_complex
        self.reversed_eval = params.reversed_eval
        self.euclidian_metric = params.euclidian_metric
        self.auxiliary_task = params.auxiliary_task
        self.tau = params.tau
        self.gramian_norm1 = params.gramian_norm1
        self.gramian_tolerance = params.gramian_tolerance

        self.min_expr_len_factor_cspeed = params.min_expr_len_factor_cspeed
        self.max_expr_len_factor_cspeed = params.max_expr_len_factor_cspeed

        self.custom_unary_probs = params.custom_unary_probs
        self.prob_trigs = params.prob_trigs
        self.prob_arc_trigs = params.prob_arc_trigs
        self.prob_logs = params.prob_logs
        self.prob_others = 1.0 - self.prob_trigs - self.prob_arc_trigs - self.prob_logs
        assert self.prob_others >= 0.0

        self.prob_int = params.prob_int
        self.precision = params.precision
        self.jacobian_precision = params.jacobian_precision

        self.max_len = params.max_len
        self.eval_value = params.eval_value
        self.skip_zero_gradient = params.skip_zero_gradient
        self.prob_positive = params.prob_positive

        self.np_positive = np.zeros(self.max_degree + 1, dtype=int)
        self.np_total = np.zeros(self.max_degree + 1, dtype=int)
        self.complex_input = "fourier" in params.tasks

        self.SYMPY_OPERATORS = {
            # Elementary functions
            sp.Add: "+",
            sp.Mul: "*",
            sp.Pow: "^",
            sp.exp: "exp",
            sp.log: "ln",
            # sp.Abs: 'abs',
            # sp.sign: 'sign',
            # Trigonometric Functions
            sp.sin: "sin",
            sp.cos: "cos",
            sp.tan: "tan",
            # sp.cot: 'cot',
            # sp.sec: 'sec',
            # sp.csc: 'csc',
            # Trigonometric Inverses
            sp.asin: "asin",
            sp.acos: "acos",
            sp.atan: "atan",
            # sp.acot: 'acot',
            # sp.asec: 'asec',
            # sp.acsc: 'acsc',
            sp.DiracDelta: "delta0",
        }

        self.operators_conv = {
            "+": 2,
            "-": 2,
            "*": 2,
            "/": 2,
            "sqrt": 1,
            "exp": 1,
            "ln": 1,
            "sin": 1,
            "cos": 1,
            "tan": 1,
            "asin": 1,
            "acos": 1,
            "atan": 1,
        }

        self.trig_ops = ["sin", "cos", "tan"]
        self.arctrig_ops = ["asin", "acos", "atan"]
        self.exp_ops = ["exp", "ln"]
        self.other_ops = ["sqrt"]

        self.operators_lyap = {
            "+": 2,
            "-": 2,
            "*": 2,
            "/": 2,
            "^": 2,
            "sqrt": 1,
            "exp": 1,
            "ln": 1,
            "sin": 1,
            "cos": 1,
            "tan": 1,
            "asin": 1,
            "acos": 1,
            "atan": 1,
            "delta0": 1,
        }

        self.operators = (
            self.operators_lyap if "fourier" in params.tasks else self.operators_conv
        )
        self.unaries = [o for o in self.operators.keys() if self.operators[o] == 1]
        self.binaries = [o for o in self.operators.keys() if self.operators[o] == 2]
        self.unary = len(self.unaries) > 0
        self.predict_bounds = params.predict_bounds

        assert self.max_int >= 1
        assert self.precision >= 2

        # variables
        self.variables = OrderedDict(
            {f"x{i}": sp.Symbol(f"x{i}") for i in range(2 * self.max_degree)}
        )

        self.eval_point = OrderedDict(
            {
                self.variables[f"x{i}"]: self.eval_value
                for i in range(2 * self.max_degree)
            }
        )

        # symbols / elements
        self.constants = ["pi", "E"]

        self.symbols = ["I", "INT+", "INT-", "FLOAT+", "FLOAT-", ".", "10^"]
        self.elements = [str(i) for i in range(10)]

        # SymPy elements
        self.local_dict = {}
        for k, v in list(self.variables.items()):
            assert k not in self.local_dict
            self.local_dict[k] = v

        # vocabulary
        self.words = (
            SPECIAL_WORDS
            + self.constants
            + list(self.variables.keys())
            + list(self.operators.keys())
            + self.symbols
            + self.elements
        )
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)
        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1
        self.func_separator = "<SPECIAL_3>"  # separate equations in a system
        self.line_separator = "<SPECIAL_4>"  # separate lines in a matrix
        self.list_separator = "<SPECIAL_5>"  # separate elements in a list
        self.mtrx_separator = "<SPECIAL_6>"  # end of a matrix
        self.neg_inf = "<SPECIAL_7>"  # negative infinity
        self.pos_inf = "<SPECIAL_8>"  # positive infinity
        logger.info(f"words: {self.word2id}")

        # initialize distribution for binary and unary-binary trees
        # self.max_ops + 1 should be enough
        self.distrib = self.generate_dist(2 * self.max_ops)

    def get_integer(self, cplex=False):
        if cplex:
            i1 = self.rng.randint(1, 100000) / 100000
            sign = 1 if self.rng.randint(2) == 0 else -1
            e = self.rng.randint(2)
            if e == 0:
                return i1 * sign
            else:
                return complex(0.0, i1 * sign)
            # i2 = self.rng.randint(1, 100000) / 100000
            # sign2 = 1 if self.rng.randint(2) == 0 else -1
            # return complex(i1 * sign, i2 * sign2)

        if self.positive and self.nonnull:
            return self.rng.randint(1, self.max_int + 1)
        if self.positive:
            return self.rng.randint(0, self.max_int + 1)
        if self.nonnull:
            s = self.rng.randint(1, 2 * self.max_int + 1)
            return s if s <= self.max_int else (self.max_int - s)

        return self.rng.randint(-self.max_int, self.max_int + 1)

    def generate_leaf(self, degree, index):
        if self.rng.rand() < self.prob_int:
            return self.get_integer()
        elif degree == 1:
            return self.variables[f"x{index}"]
        else:
            return self.variables[f"x{self.rng.randint(degree)}"]

    def generate_ops(self, arity):
        if arity == 1:
            if self.custom_unary_probs:
                w = [
                    self.prob_trigs,
                    self.prob_arc_trigs,
                    self.prob_logs,
                    self.prob_others,
                ]
                s = [self.trig_ops, self.arctrig_ops, self.exp_ops, self.other_ops]
                return self.rng.choice(s, p=w)
            else:
                return self.rng.choice(self.unaries)

        else:
            return self.rng.choice(self.binaries)

    def generate_dist(self, max_ops):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible unary-binary trees
        that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(n, 0) = 0
            D(0, e) = 1
            D(n, e) = D(n, e - 1) + p_1 * D(n- 1, e) + D(n - 1, e + 1)
        p1 =  if binary trees, 1 if unary binary
        """
        p1 = 1 if self.unary else 0
        # enumerate possible trees
        D = []
        D.append([0] + ([1 for i in range(1, 2 * max_ops + 1)]))
        for n in range(1, 2 * max_ops + 1):  # number of operators
            s = [0]
            for e in range(1, 2 * max_ops - n + 1):  # number of empty nodes
                s.append(s[e - 1] + p1 * D[n - 1][e] + D[n - 1][e + 1])
            D.append(s)
        assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
        return D

    def sample_next_pos(self, nb_empty, nb_ops):
        """
        Sample the position of the next node (binary case).
        Sample a position in {0, ..., `nb_empty` - 1}.
        """
        assert nb_empty > 0
        assert nb_ops > 0
        probs = []
        if self.unary:
            for i in range(nb_empty):
                probs.append(self.distrib[nb_ops - 1][nb_empty - i])
        for i in range(nb_empty):
            probs.append(self.distrib[nb_ops - 1][nb_empty - i + 1])
        probs = [p / self.distrib[nb_ops][nb_empty] for p in probs]
        probs = np.array(probs, dtype=np.float64)
        e = self.rng.choice(len(probs), p=probs)
        arity = 1 if self.unary and e < nb_empty else 2
        e %= nb_empty
        return e, arity

    def generate_tree(self, nb_ops, degree, index=0):
        tree = Node(0)
        empty_nodes = [tree]
        next_en = 0
        nb_empty = 1
        while nb_ops > 0:
            next_pos, arity = self.sample_next_pos(nb_empty, nb_ops)
            for n in empty_nodes[next_en : next_en + next_pos]:
                n.value = self.generate_leaf(degree, index)
            next_en += next_pos
            empty_nodes[next_en].value = self.generate_ops(arity)
            for _ in range(arity):
                e = Node(0)
                empty_nodes[next_en].push_child(e)
                empty_nodes.append(e)
            nb_empty += arity - 1 - next_pos
            nb_ops -= 1
            next_en += 1
        for n in empty_nodes[next_en:]:
            n.value = self.generate_leaf(degree, index)
        return tree

    def generate_polynomial(
        self, nterm, max_factor, degree, unaries, noconstant=True, complex_coeffs=False
    ):
        pol = set()
        for i in range(nterm):
            nfactor = self.rng.randint(1, max_factor + 1)
            vars = set()
            for j in range(nfactor):
                vars.add(
                    (self.rng.randint(0, degree), self.rng.randint(0, len(unaries)))
                )
            pol.add(tuple(vars))
        for i in range(len(pol)):
            v = list(pol)[i]
            for j in range(len(v)):
                op = unaries[v[j][1]]
                var = Node(self.variables[f"x{v[j][0]}"])
                if op == "id":
                    term = var
                elif op == "ln":
                    term = Node("ln", [Node("+", [Node(1), var])])
                elif len(op) > 3 and op[:3] == "pow":
                    term = Node("^", [var, Node(int(op[3:]))])
                else:
                    term = Node(op, [var])
                p = term if j == 0 else Node("*", [p, term])
            coeff = self.get_integer(complex_coeffs)
            if complex_coeffs:
                p = Node("*", [Node(coeff), p])
                tree = p if i == 0 else Node("+", [tree, p])
            else:
                if abs(coeff) != 1:
                    p = Node("*", [Node(abs(coeff)), p])
                tree = p if i == 0 else Node("+" if coeff > 0 else "-", [tree, p])
        if not noconstant:
            coeff = self.get_integer(complex_coeffs)
            if complex_coeffs:
                tree = Node("+", [tree, Node(coeff)])
            else:
                tree = Node("+" if coeff > 0 else "-", [tree, Node(abs(coeff))])
        return tree

    def batch_sequences(self, sequences):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sequences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(
            self.pad_index
        )
        assert lengths.min().item() > 2

        sent[0] = self.eos_index
        for i, s in enumerate(sequences):
            sent[1 : lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths

    def write_int(self, val):
        """
        Convert a decimal integer to a representation in base 10.
        """
        res = []
        neg = val < 0
        val = -val if neg else val
        while True:
            rem = val % 10
            val = val // 10
            res.append(str(rem))
            if val == 0:
                break
        res.append("INT-" if neg else "INT+")
        return res[::-1]

    def parse_int(self, lst):
        """
        Parse a list that starts with an integer.
        Return the integer value, and the position it ends in the list.
        """
        if len(lst) < 2 or lst[0] not in ["INT+", "INT-"] or not lst[1].isdigit():
            raise InvalidPrefixExpression("Invalid integer in prefix expression")
        val = int(lst[1])
        i = 1
        for x in lst[2:]:
            if not x.isdigit():
                break
            val = val * 10 + int(x)
            i += 1
        if lst[0] == "INT-":
            val = -val
        return val, i + 1

    def write_float(self, value, precision=None):
        """
        Write a float number.
        """
        precision = self.precision if precision is None else precision
        assert value not in [-np.inf, np.inf]
        res = ["FLOAT+"] if value >= 0.0 else ["FLOAT-"]
        m, e = (f"%.{precision}e" % abs(value)).split("e")
        assert e[0] in ["+", "-"]
        e = int(e[1:] if e[0] == "+" else e)
        return res + list(m) + ["10^"] + self.write_int(e)

    def parse_float(self, lst):
        """
        Parse a list that starts with a float.
        Return the float value, and the position it ends in the list.
        """
        if len(lst) < 2 or lst[0] not in ["FLOAT+", "FLOAT-"]:
            return np.nan, 0
        sign = -1 if lst[0] == "FLOAT-" else 1
        if not lst[1].isdigit():
            return np.nan, 1
        mant = 0.0
        i = 1
        for x in lst[1:]:
            if not (x.isdigit()):
                break
            mant = mant * 10.0 + int(x)
            i += 1
        if len(lst) > i and lst[i] == ".":
            i += 1
            mul = 0.1
            for x in lst[i:]:
                if not (x.isdigit()):
                    break
                mant += mul * int(x)
                mul *= 0.1
                i += 1
        mant *= sign
        if len(lst) > i and lst[i] == "10^":
            i += 1
            try:
                exp, offset = self.parse_int(lst[i:])
            except InvalidPrefixExpression:
                return np.nan, i
            i += offset
        else:
            exp = 0
        return mant * (10.0 ** exp), i

    def write_complex(self, value, precision=None):
        """
        Write a complex number.
        """
        if value == 0:
            return self.write_float(0, precision)
        res = []
        if value.imag != 0:
            res = self.write_float(value.imag, precision) + ["I"]
        if value.real != 0:
            res = res + self.write_float(value.real, precision)
        return res

    def parse_complex(self, lst):
        """
        Parse a list that starts with a complex number.
        Return the complex value, and the position it ends in the list.
        """
        first_val, len1 = self.parse_float(lst)
        if np.isnan(first_val):
            return np.nan, len1
        if len(lst) <= len1 or lst[len1] != "I":
            return first_val, len1
        second_val, len2 = self.parse_float(lst[len1 + 1 :])
        if np.isnan(second_val):
            return complex(0, first_val), len1 + 1
        return complex(second_val, first_val), len1 + 1 + len2

    def input_to_infix(self, lst):
        res = ""
        degree, offset = self.parse_int(lst)
        res = str(degree) + "|"

        offset += 1
        l1 = lst[offset:]
        if self.complex_input:
            nr_eqs = 1
        else:
            nr_eqs = degree
        for i in range(nr_eqs):
            s, l2 = self.prefix_to_infix(l1)
            res = res + s + "|"
            l1 = l2[1:]
        return res[:-1]

    def output_to_infix(self, lst):
        val, _ = self.parse_float(lst)
        return str(val)

    def prefix_to_infix(self, expr):
        """
        Parse an expression in prefix mode, and output it in either:
          - infix mode (returns human readable string)
          - develop mode (returns a dictionary with the simplified expression)
        """
        cplx = self.complex_input
        if len(expr) == 0:
            raise InvalidPrefixExpression("Empty prefix list.")
        t = expr[0]
        if t in self.operators.keys():
            args = []
            l1 = expr[1:]
            for _ in range(self.operators[t]):
                i1, l1 = self.prefix_to_infix(l1)
                args.append(i1)
            if self.operators[t] == 1:
                return f"{t}({args[0]})", l1
            return f"({args[0]}{t}{args[1]})", l1
            # return f'({args[0]}){t}({args[1]})', l1
        elif t in self.variables or t in self.constants or t == "I":
            return t, expr[1:]
        elif t == "FLOAT+" or t == "FLOAT-":
            if cplx:
                val, i = self.parse_complex(expr)
            else:
                val, i = self.parse_float(expr)
        else:
            val, i = self.parse_int(expr)
        return str(val), expr[i:]

    def _sympy_to_prefix(self, op, expr):
        """
        Parse a SymPy expression given an initial root operator.
        """
        n_args = len(expr.args)

        assert (
            (op == "+" or op == "*")
            and (n_args >= 2)
            or (op != "+" and op != "*")
            and (1 <= n_args <= 2)
        )

        # square root
        if (
            op == "^"
            and isinstance(expr.args[1], sp.Rational)
            and expr.args[1].p == 1
            and expr.args[1].q == 2
        ):
            return ["sqrt"] + self.sympy_to_prefix(expr.args[0])

        # parse children
        parse_list = []
        for i in range(n_args):
            if i == 0 or i < n_args - 1:
                parse_list.append(op)
            parse_list += self.sympy_to_prefix(expr.args[i])

        return parse_list

    def sympy_to_prefix(self, expr):
        """
        Convert a SymPy expression to a prefix one.
        """
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        elif isinstance(expr, sp.Integer):
            return self.write_int(int(str(expr)))
        elif isinstance(expr, sp.Float):
            return self.write_float(float(str(expr)))
        elif isinstance(expr, sp.Rational):
            return ["/"] + self.write_int(int(expr.p)) + self.write_int(int(expr.q))
        elif expr == sp.E:
            return ["E"]
        elif expr == sp.pi:
            return ["pi"]
        elif expr == sp.I:
            raise UnknownSymPyOperator(f"Unknown SymPy operator: {expr}")

        # SymPy operator
        for op_type, op_name in self.SYMPY_OPERATORS.items():
            if isinstance(expr, op_type):
                return self._sympy_to_prefix(op_name, expr)
        # unknown operator
        raise UnknownSymPyOperator(f"Unknown SymPy operator: {expr}")

    def encode_expr(self, tree, cplx=False):
        pref = tree.prefix().split(", ")
        res = []
        for p in pref:
            if (p.startswith("-") and p[1:].isdigit()) or p.isdigit():
                res.extend(self.write_int(int(p)))
            elif cplx and (
                (p.startswith("-") and p[1:2].isdigit())
                or p.startswith("(")
                or p[0:1].isdigit()
            ):
                res.extend(self.write_complex(complex(p)))
            else:
                res.append(p)
        return res

    @timeout(5)
    def compute_gradient(self, expr, point, degree):
        values = np.zeros(degree, dtype=complex)
        try:
            for i in range(degree):
                grad = expr.diff(self.variables[f"x{i}"])
                values[i] = grad.subs(point).evalf()
        except TimeoutError:
            raise
        except Exception:
            raise
        return values

    def gen_ode_system_convergence(self, return_system=False):
        """
        Generate systems of functions, and the corresponding convergence speed in zero.
        Start by generating a random system S, use SymPy to compute formal jacobian
        and evaluate it in zero, find largest eigenvalue
        Encode this as a prefix sensence
        """
        degree = self.rng.randint(self.min_degree, self.max_degree + 1)
        nb_ops = self.rng.randint(
            self.min_expr_len_factor_cspeed * degree + 3,
            self.max_expr_len_factor_cspeed * degree + 3,
            size=(degree,),
        )

        while True:
            system = []
            i = 0
            ngen = 0
            while i < degree:
                # generate expression
                expr = self.generate_tree(nb_ops[i], degree)
                ngen += 1
                # sympy zone
                try:
                    expr_sp = sp.S(expr, locals=self.local_dict)
                    # skip constant or invalid expressions
                    if len(expr_sp.free_symbols) == 0 or has_inf_nan(expr_sp):
                        continue
                    # evaluate gradient in point
                    values = self.compute_gradient(expr_sp, self.eval_point, degree)
                    if np.isnan(values).any() or np.isinf(values).any():
                        continue
                    if self.skip_zero_gradient and not values.any():
                        continue
                except TimeoutError:
                    continue
                except (ValueError, TypeError):
                    continue
                except Exception as e:
                    logger.error(
                        "An unknown exception of type {0} occurred in line {1} "
                        'for expression "{2}". Arguments:{3!r}.'.format(
                            type(e).__name__,
                            sys.exc_info()[-1].tb_lineno,
                            expr_sp,
                            e.args,
                        )
                    )
                    continue

                system.append(expr)
                if i == 0:
                    jacobian = values
                else:
                    jacobian = np.vstack((jacobian, values))
                i += 1
            if self.skip_zero_gradient:
                skip = False
                for i in range(degree):
                    if not jacobian[:, [i]].any():
                        skip = True
                        break
                if skip:
                    continue

            cspeed = -max(np.linalg.eigvals(jacobian).real)

            if self.prob_positive == 0 and cspeed > 0:
                continue
            if self.prob_positive == 1 and cspeed <= 0:
                continue
            if (
                self.prob_positive > 0
                and self.prob_positive < 1
                and self.np_total[degree] > 10
            ):
                proportion = self.np_positive[degree] / self.np_total[degree]
                if cspeed > 0 and proportion > self.prob_positive:
                    continue
                if cspeed <= 0 and proportion < self.prob_positive:
                    continue

            self.np_total[degree] += 1
            if cspeed > 0:
                self.np_positive[degree] += 1
            break

        # # debug
        # logger.info(str(cspeed))
        # logger.info(str(cspeed) + "\t" + " ||||| ".join(str(s) for s in system[:3]))
        # print(degree, str(ngen) + " : " + str((ngen - degree) / ngen * 100.0))

        # encode input
        x = self.write_int(degree)
        for s in system:
            x.append(self.func_separator)
            x.extend(self.encode_expr(s))

        # encode output: eigenvalue, and optionally the Jacobian matrix
        eigenvalue = self.write_float(cspeed)
        if self.predict_jacobian:
            y = []
            for row in jacobian:
                for value in row:
                    y.extend(
                        self.write_complex(value, precision=self.jacobian_precision)
                    )
                    y.append(self.list_separator)
                y.append(self.line_separator)
            y.append(self.mtrx_separator)
            y.extend(eigenvalue)
        else:
            y = eigenvalue

        if return_system:
            return x, y, system
        else:
            return x, y

    @timeout(5)
    def compute_gradient_control(self, expr, point, degree, p):
        if self.allow_complex:
            A = np.zeros(degree, dtype=complex)
            B = np.zeros(p, dtype=complex)
        else:
            A = np.zeros(degree, dtype=float)
            B = np.zeros(p, dtype=float)
        try:
            for i in range(degree + p):
                grad = expr.diff(self.variables[f"x{i}"])
                val = grad.subs(point).evalf()
                if i < degree:
                    A[i] = val
                else:
                    B[i - degree] = val
        except TimeoutError:
            raise
        except Exception:
            raise
        return A, B

    def gen_control(self, return_system=False, skip_unstable=False):
        """
        Generate systems of functions, data for controlability
        """
        degree = self.rng.randint(self.min_degree, self.max_degree + 1)
        p = self.rng.randint(1, degree // 2 + 1)
        nb_ops = self.rng.randint(degree + p, 2 * (degree + p) + 3, size=(degree,))
        while True:
            system = []
            i = 0
            ngen = 0
            while i < degree:
                # generate expression
                expr = self.generate_tree(
                    nb_ops[i], degree + p
                )  # si tau>0 doit on garantir l'existence de t (x{degree + p})?
                ngen += 1
                # sympy zone
                try:
                    expr_sp = sp.S(expr, locals=self.local_dict)
                    # skip constant or invalid expressions
                    if len(expr_sp.free_symbols) == 0 or has_inf_nan(expr_sp):
                        continue
                    # evaluate gradient in point
                    valA, valB = self.compute_gradient_control(
                        expr_sp, self.eval_point, degree, p
                    )
                    if (
                        np.isnan(valA).any()
                        or np.isinf(valA).any()
                        or np.isnan(valB).any()
                        or np.isinf(valB).any()
                    ):
                        continue
                    if self.skip_zero_gradient and not valA.any():
                        continue
                except TimeoutError:
                    continue
                except (ValueError, TypeError):
                    continue
                except Exception as e:
                    logger.error(
                        "An unknown exception of type {0} occurred in line {1} "
                        'for expression "{2}". Arguments:{3!r}.'.format(
                            type(e).__name__,
                            sys.exc_info()[-1].tb_lineno,
                            expr_sp,
                            e.args,
                        )
                    )
                    continue

                system.append(expr)
                if i == 0:
                    A = valA
                    B = valB
                else:
                    A = np.vstack((A, valA))
                    B = np.vstack((B, valB))
                i += 1
            if self.skip_zero_gradient:
                skip = False
                for i in range(degree):
                    if not A[:, [i]].any():
                        skip = True
                        break
                for i in range(p):
                    if not B[:, [i]].any():
                        skip = True
                        break
                if skip:
                    continue
            try:
                C = ctrl.ctrb(A, B)
                d = degree - np.linalg.matrix_rank(C, 1.0e-6)
                if d != 0 and (skip_unstable or self.prob_positive > 0.0):
                    continue
                if self.predict_gramian and d == 0:
                    # C = ctrl.lyap(A, - B @ B.T)
                    # K = - B.T @ np.linalg.inv(C)
                    A = A / np.linalg.norm(A)
                    B = B / np.linalg.norm(A)
                    tau = 1
                    yint = []
                    # We want to integrate a matrix over [0,tau]
                    # and all the integrate functions I found are for scalars.
                    # So we do it term by term
                    for i in range(degree):  # divide in row
                        yint_line = []
                        for j in range(degree):  # divide in column

                            dt = np.linspace(
                                0, tau, num=40
                            )  # integration path [0,tau] and 40 points
                            yint0 = []
                            for k in range(len(dt)):
                                # vector i with the component to be integrated (i,j),
                                # evaluated at each point of the integration path
                                res = (
                                    (expm(A * (tau - dt[k])))
                                    @ (B @ B.T)
                                    @ (expm(A.T * (tau - dt[k])))
                                )[i]
                                yint0.append(
                                    res[j]
                                )  # vector of the component (i,j) along itegration path
                            resline = (cumtrapz(yint0, dt, initial=0))[
                                len(dt) - 1
                            ]  # integration with cumulative trapezz
                            yint_line.append(resline)  # reconstruct the line
                        yint.append(yint_line)  # reconstruct the matrix
                    if np.isnan(yint).any() or np.isinf(yint).any():
                        continue
                    Ctau = (
                        expm(-tau * A) @ np.array(yint) @ expm(-tau * A.T)
                    )  # From the gramian to the true C
                    if np.isnan(Ctau).any() or np.isinf(Ctau).any():
                        continue
                    K = -B.T @ (np.linalg.inv(Ctau + 1e-6 * np.eye(degree)))
                    if np.isnan(K).any() or np.isinf(K).any():
                        continue

                    with np.nditer(K, op_flags=["readwrite"]) as it:
                        for x in it:
                            x[...] = float(f"%.{self.jacobian_precision}e" % x)

                    if max(np.linalg.eigvals(A + B @ K).real) > 0:
                        # Check that A+B@K is stable, which is equivalent to
                        # check_gramian
                        # print("UNSTABLE")
                        continue

            except Exception:
                # logger.error("An unknown exception of type {0} occurred
                # in line {1} for expression \"{2}\". Arguments:{3!r}.".format(
                # type(e).__name__, sys.exc_info()[-1].tb_lineno, expr_sp, e.args))
                continue
            break
        # # debug
        # logger.info(str(cspeed))
        # logger.info(str(cspeed) + "\t" + " ||||| ".join(str(s) for s in system[:3]))
        # print(degree, str(ngen) + " : " + str((ngen - degree) / ngen * 100.0))

        # encode input
        x = self.write_int(degree)
        for s in system:
            x.append(self.func_separator)
            x.extend(self.encode_expr(s))

        # encode output: dimension of control subspace and optionally the Gramian matrix
        if self.qualitative:
            controlable = 1 if d == 0 else 0
            y = self.write_int(controlable)
        else:
            y = self.write_int(d)
            if self.predict_gramian and d == 0:
                K = np.array(K)
                y.append(self.mtrx_separator)
                for row in K:
                    for value in row:
                        y.extend(self.write_complex(value, self.jacobian_precision))
                        y.append(self.list_separator)
                    y.append(self.line_separator)

        if self.max_len > 0 and (len(x) >= self.max_len or len(y) >= self.max_len):
            return None

        if return_system:
            return x, y, system, p
        else:
            return x, y

    @timeout(5)
    def compute_gradient_control_t(self, expr, point, degree, p):
        A = []
        B = []
        try:
            for i in range(degree + p):
                grad = expr.diff(self.variables[f"x{i}"])
                val = grad.subs(point).evalf()
                val = simplify(val, 2)
                if i < degree:
                    A.append(val)
                else:
                    B.append(val)
        except TimeoutError:
            raise
        except Exception:
            raise
        return A, B

    @timeout(10)
    def compute_rank(self, A, B, degree, p, val):
        Bi = B
        for i in range(1, int(val * degree / p) + 1):
            E = B.diff(self.variables[f"x{degree + p}"])
            B = E - A * B
            Bi = Bi.row_join(B)
        d = 1
        for i in range(5):
            value = (i + 1) * self.tau / 5 - 0.01
            # D = w(value)
            D = Bi.subs({self.variables[f"x{degree + p}"]: value})
            D = np.array(D).astype(np.complex)
            if np.isnan(D).any() or np.isinf(D).any():
                continue
            d = degree - np.linalg.matrix_rank(D, 1.0e-6)
            if d == 0:
                break
        return d

    # @timeout(20)
    def gen_control_t(self):
        """
        Generate systems of functions, data for controlability
        """
        while True:
            degree = self.rng.randint(self.min_degree, self.max_degree + 1)
            p = self.rng.randint(1, degree // 2 + 1)
            nb_ops = self.rng.randint(degree + p, 2 * (degree + p) + 3, size=(degree,))
            ev_point = OrderedDict(
                {self.variables[f"x{i}"]: self.eval_value for i in range(degree + p)}
            )
            system = []
            i = 0
            A = sp.Matrix()
            B = sp.Matrix()
            ngen = 0
            while i < degree:
                # generate expression
                # si tau>0 doit on garantir l'existence de t (x{degree + p}) ?
                expr = self.generate_tree(nb_ops[i], degree + p + 1)
                ngen += 1
                # sympy zone
                try:
                    expr_sp = sp.S(expr, locals=self.local_dict)
                    # skip constant or invalid expressions
                    if len(expr_sp.free_symbols) == 0 or has_inf_nan(expr_sp):
                        continue
                    # evaluate gradient in point
                    valA, valB = self.compute_gradient_control_t(
                        expr_sp, ev_point, degree, p
                    )
                    # print('valA', valA)
                    # print('valB', valB)
                    if any(has_inf_nan(a) for a in valA) or any(
                        has_inf_nan(a) for a in valB
                    ):
                        continue
                    if self.skip_zero_gradient and all(a == 0 for a in valA):
                        continue
                except TimeoutError:
                    continue
                except (ValueError, TypeError):
                    continue
                except Exception as e:
                    logger.error(
                        "An unknown exception of type {0} occurred in line {1} "
                        'for expression "{2}". '
                        "Arguments:{3!r}.".format(
                            type(e).__name__,
                            sys.exc_info()[-1].tb_lineno,
                            expr_sp,
                            e.args,
                        )
                    )
                    continue

                system.append(expr)
                v1 = sp.Matrix(1, degree, valA)
                v2 = sp.Matrix(1, p, valB)
                A = A.col_join(v1)
                B = B.col_join(v2)
                i += 1

            if self.skip_zero_gradient:
                if any(all(A[j, i] == 0 for j in range(degree)) for i in range(degree)):
                    continue
                if any(all(B[j, i] == 0 for j in range(degree)) for i in range(p)):
                    continue

            try:
                d = self.compute_rank(A, B, degree, p, 2)
            except TimeoutError:
                continue
            # except FloatingPointError:
            #     continue
            except Exception as e:
                logger.error(
                    "An unknown exception of type {0} occurred in line {1} "
                    'for expression "{2}". '
                    "Arguments:{3!r}.".format(
                        type(e).__name__, sys.exc_info()[-1].tb_lineno, expr_sp, e.args
                    )
                )
                continue
            break
        # # debug
        # logger.info(str(cspeed))
        # logger.info(str(cspeed) + "\t" + " ||||| ".join(str(s) for s in system[:3]))
        # print(degree, str(ngen) + " : " + str((ngen - degree) / ngen * 100.0))

        # print(', '.join(f"{s} {t:.3f}" for s, t in times))

        # encode input
        x = self.write_int(degree)
        for s in system:
            x.append(self.func_separator)
            x.extend(self.encode_expr(s))

        # encode output: dimension of control subspace and optionally the Gramian matrix
        controlable = 1 if d == 0 else 0
        y = self.write_int(controlable)

        if self.max_len > 0 and (len(x) >= self.max_len or len(y) >= self.max_len):
            return None

        return x, y

    def generate_cond_init(self, max_delay, dimension, unariesexp, unariesfk):
        pol = set()
        nfactor = self.rng.randint(1, max_delay + 1)
        # print(nfactor)
        delay = np.zeros(dimension)
        bounds = []
        vars = set()
        for j in range(nfactor):
            vars.add(
                (self.rng.randint(0, dimension), self.rng.randint(0, len(unariesexp)))
            )
        pol.add(tuple(vars))
        # print(pol)
        for i in range(len(pol)):
            v = list(pol)[i]
            # print(len(v))
            # print(v[len(v)-1])
            # print(v[len(v)-1][0])
            # print(v[0][0])
            # print(delay)
            for j in range(len(v)):
                op = unariesexp[v[j][1]]
                var = Node(self.variables[f"x{v[j][0]}"])
                if op == "id":
                    term = var
                elif len(op) > 3 and op[:3] == "pow":
                    term = Node("^", [var, Node(int(op[3:]))])
                elif op == "expi":
                    a_d = self.rng.randint(-100, 100)
                    # b = self.rng.randint(-100, 100)#Not needed for now
                    b_d = 0
                    term = Node(
                        "exp",
                        [
                            Node(
                                "+",
                                [
                                    Node("*", [Node(a_d), Node("*", [Node("I"), var])]),
                                    Node(b_d),
                                ],
                            )
                        ],
                    )
                    delay[v[j][0]] = delay[v[j][0]] + a_d
                    # print(delay[v[j][0]])
                else:
                    term = Node(op, [var])
                p = term if j == 0 else Node("*", [p, term])
        expr_delay = p
        # print(sp.S(expr_delay))
        for i in range(dimension):
            k = self.rng.randint(0, len(unariesfk))
            op = unariesfk[k]
            var = Node(self.variables[f"x{i}"])
            a = self.rng.randint(-100, 100)
            # b = self.rng.randint(-100, 100)
            # inclure b plus tard not needed now avec les delays
            b = 0
            var = Node("+", [Node("*", [Node(a), var]), Node(b)])
            if op == "sinc":
                bounds.append(
                    [-abs(a) / (2 * np.pi), abs(a) / (2 * np.pi)]
                )  # fouriertiser
                term = Node("/", [Node("sin", [var]), var])
                # print(sp.S(term))
            elif op == "1":
                bounds.append([0, 0])
                term = Node(1)
            elif op == "delta0":
                bounds.append([-np.inf, np.inf])
                term = Node(op, [var])
            elif op == "gauss":
                bounds.append([-np.inf, np.inf])
                term = Node(
                    "exp", [Node("*", [Node(-1), Node("^", [var, Node(2)])])]
                )  # checker
            else:
                return None
            # Message d'erreur
            # print(sp.S(term))
            p = term if i == 0 else Node("*", [p, term])
            bounds[i][0] = bounds[i][0] + delay[i] / (2 * np.pi)
            bounds[i][1] = bounds[i][1] + delay[i] / (2 * np.pi)
            # print(delay[i])
        u0 = Node("*", [expr_delay, p])
        # u0f = Node('*', [exprf, pf])

        return u0, bounds

    def gen_fourier_cond_init(self):
        while True:
            try:
                dimension = self.rng.randint(self.min_degree, self.max_degree + 1)
                nb_ops = self.rng.randint(dimension, 2 * dimension + 3)
                # Generate differential operator
                unariesd = ["id", "pow2", "pow4"]
                expr = self.generate_polynomial(
                    nb_ops, 4, dimension, unariesd, True, False
                )
                # print(sp.S(expr))
                # Fourier transform of the differential operator
                PF = OrderedDict(
                    {
                        self.variables[f"x{i}"]: 2
                        * np.pi
                        * 1j
                        * self.variables[f"x{i}"]
                        for i in range(self.max_degree)
                    }
                )
                poly_fourier = sp.S(expr).subs(PF)
                # print(poly_fourier)
                # Generate initial condition
                unariesexp = ["expi"]
                unariesfk = ["1", "sinc", "delta0", "gauss"]
                max_delay_op = 2 * dimension
                expr_u0, bounds = self.generate_cond_init(
                    max_delay_op, dimension, unariesexp, unariesfk
                )
                # print(sp.S(expr_u0))
                # print(bounds)
                # Minimization of the Fourier transform of the differential operator
                # on the frequency of the initial conditions
                dum_point = np.zeros(dimension, dtype=float) + 0.5
                max_f = opt.minimize(
                    expr_to_fun_real,
                    dum_point,
                    args=(poly_fourier, dimension),
                    method="TNC",
                    bounds=bounds,
                    options={"ftol": 1e-15, "gtol": 1e-15},
                )
                # print(max_f.fun)
                if not max_f.success:
                    # logger.info(f'optimization error')
                    continue
                if max_f.fun < -1e14:
                    reg = 0  # -1
                    stab = 0
                elif max_f.fun < 0:
                    reg = 1  # 0
                    stab = 0
                elif max_f.fun >= 0:
                    reg = 1
                    stab = 1
                else:
                    # logger.info(f'optimization error in value')
                    continue
            except Exception as e:
                print(e)
                continue
            break

        # encode input
        x = self.write_int(dimension)
        x.append(self.func_separator)
        x.extend(self.encode_expr(expr, True))
        x.append(self.func_separator)
        x.extend(self.encode_expr(expr_u0, True))

        # encode output
        y = self.write_int(reg)
        y.append(self.func_separator)
        y.extend(self.write_int(stab))
        if self.predict_bounds:
            y.append(self.func_separator)
            for i in range(len(bounds)):
                if bounds[i][0] == np.inf:
                    y.append(self.pos_inf)
                elif bounds[i][0] == -np.inf:
                    y.append(self.neg_inf)
                else:
                    y.extend(self.write_float(bounds[i][0], 2))
                y.append(self.list_separator)
                if bounds[i][1] == np.inf:
                    y.append(self.pos_inf)
                elif bounds[i][1] == -np.inf:
                    y.append(self.neg_inf)
                else:
                    y.extend(self.write_float(bounds[i][1], 2))
                y.append(self.line_separator)

        return x, y

    def create_train_iterator(self, task, data_path, params):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating train iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=True,
            params=params,
            path=(None if data_path is None else data_path[task][0]),
        )
        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 1800),
            batch_size=params.batch_size,
            num_workers=(
                params.num_workers
                if data_path is None or params.num_workers == 0
                else 1
            ),
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    def create_test_iterator(
        self, data_type, task, data_path, batch_size, params, size
    ):
        """
        Create a dataset for this environment.
        """
        assert data_type in ["valid", "test"]
        logger.info(f"Creating {data_type} iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=False,
            params=params,
            path=(
                None
                if data_path is None
                else data_path[task][1 if data_type == "valid" else 2]
            ),
            size=size,
        )
        return DataLoader(
            dataset,
            timeout=0,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument(
            "--max_int", type=int, default=10, help="Maximum integer value"
        )
        parser.add_argument(
            "--precision", type=int, default=3, help="Float numbers precision"
        )
        parser.add_argument(
            "--jacobian_precision",
            type=int,
            default=1,
            help="Float numbers precision in the Jacobian",
        )
        parser.add_argument(
            "--positive",
            type=bool_flag,
            default=False,
            help="Do not sample negative numbers",
        )
        parser.add_argument(
            "--nonnull", type=bool_flag, default=True, help="Do not sample zeros"
        )
        parser.add_argument(
            "--predict_jacobian",
            type=bool_flag,
            default=False,
            help="Predict the Jacobian matrix",
        )
        parser.add_argument(
            "--predict_gramian",
            type=bool_flag,
            default=False,
            help="Predict the Gramian matrix",
        )
        parser.add_argument(
            "--qualitative",
            type=bool_flag,
            default=False,
            help="Binary output: system is stable or controllable",
        )
        parser.add_argument(
            "--allow_complex",
            type=bool_flag,
            default=False,
            help="Allow complex values in A and B",
        )
        parser.add_argument(
            "--reversed_eval",
            type=bool_flag,
            default=False,
            help="Validation set is dim whereas train set is test control",
        )
        parser.add_argument(
            "--euclidian_metric",
            type=bool_flag,
            default=False,
            help="Simple metric for gramian comparison",
        )
        parser.add_argument(
            "--auxiliary_task",
            type=bool_flag,
            default=False,
            help="Gramian as auxiliary task",
        )
        parser.add_argument(
            "--tau", type=int, default=0, help="if > 0 time span for controllability"
        )
        parser.add_argument(
            "--gramian_norm1",
            type=bool_flag,
            default=False,
            help="Use norm1 as Euclidian distance for Gramian",
        )
        parser.add_argument(
            "--gramian_tolerance",
            type=float,
            default=0.1,
            help="Tolerance level for Gramian euclidian distance",
        )
        parser.add_argument(
            "--predict_bounds",
            type=bool_flag,
            default=True,
            help="Predict bounds for Fourier with initial conditions",
        )

        parser.add_argument(
            "--prob_int",
            type=float,
            default=0.3,
            help="Probability of int vs variables",
        )
        parser.add_argument(
            "--min_degree",
            type=int,
            default=2,
            help="Minimum degree of ode / nb of variables",
        )
        parser.add_argument(
            "--max_degree",
            type=int,
            default=6,
            help="Maximum degree of ode / nb of variables",
        )

        parser.add_argument(
            "--min_expr_len_factor_cspeed",
            type=int,
            default=0,
            help="In cspeed, min nr of operators in system eqs: 3+k degree",
        )
        parser.add_argument(
            "--max_expr_len_factor_cspeed",
            type=int,
            default=2,
            help="In cspeed, min nr of operators in system eqs: 3+k degree",
        )

        parser.add_argument(
            "--custom_unary_probs",
            type=bool_flag,
            default=False,
            help="Lyapunov function is a polynomial",
        )
        parser.add_argument(
            "--prob_trigs",
            type=float,
            default=0.333,
            help="Probability of trig operators",
        )
        parser.add_argument(
            "--prob_arc_trigs",
            type=float,
            default=0.333,
            help="Probability of inverse trig operators",
        )
        parser.add_argument(
            "--prob_logs",
            type=float,
            default=0.222,
            help="Probability of logarithm and exponential operators",
        )

        parser.add_argument(
            "--eval_value",
            type=float,
            default=0.0,
            help="Evaluation point for all variables",
        )
        parser.add_argument(
            "--skip_zero_gradient",
            type=bool_flag,
            default=False,
            help="No gradient can be zero at evaluation point",
        )

        parser.add_argument(
            "--prob_positive",
            type=float,
            default=-1.0,
            help=(
                "Proportion of positive convergence speed "
                "(for all degrees, -1.0 = no control)"
            ),
        )

        parser.add_argument(
            "--eval_size",
            type=int,
            default=10000,
            help="Size and valid and test sample",
        )


class EnvDataset(Dataset):
    def __init__(self, env, task, train, params, path, size=None):
        super(EnvDataset).__init__()
        self.env = env
        self.train = train
        self.task = task
        self.batch_size = params.batch_size
        self.env_base_seed = params.env_base_seed
        self.path = path
        self.global_rank = params.global_rank
        self.count = 0
        assert task in ODEEnvironment.TRAINING_TASKS
        assert size is None or not self.train

        # batching
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size

        # generation, or reloading from file
        if path is not None:
            assert os.path.isfile(path)
            logger.info(f"Loading data from {path} ...")
            with io.open(path, mode="r", encoding="utf-8") as f:
                # either reload the entire file, or the first N lines
                # (for the training set)
                if not train:
                    lines = [line.rstrip().split("|") for line in f]
                else:
                    lines = []
                    for i, line in enumerate(f):
                        if i == params.reload_size:
                            break
                        if i % params.n_gpu_per_node == params.local_rank:
                            lines.append(line.rstrip().split("|"))
            self.data = [xy.split("\t") for _, xy in lines]
            self.data = [xy for xy in self.data if len(xy) == 2]
            logger.info(f"Loaded {len(self.data)} equations from the disk.")

            if task == "ode_control" and params.reversed_eval and not self.train:
                self.data = [
                    (x, "INT+ 1" if y == "INT+ 0" else "INT+ 0") for (x, y) in self.data
                ]

            if task == "ode_convergence_speed" and params.qualitative:
                self.data = [
                    (x, "INT+ 1" if y[:7] == "FLOAT- " else "INT+ 0")
                    for (x, y) in self.data
                ]

            if (
                task == "fourier_cond_init" and not params.predict_bounds
            ):  # "INT+ X <SPECIAL_3> INT+ X"
                self.data = [(x, y[:25]) for (x, y) in self.data]

            # if we are not predicting the Jacobian, remove it
            if task == "ode_convergence_speed" and not params.predict_jacobian:
                self.data = [
                    (x, y[y.index(env.mtrx_separator) + len(env.mtrx_separator) + 1 :])
                    if env.mtrx_separator in y
                    else (x, y)
                    for (x, y) in self.data
                ]

        # dataset size: infinite iterator for train,
        # finite for valid / test (default of 5000 if no file provided)
        if self.train:
            self.size = 1 << 60
        elif size is None:
            self.size = 5000 if path is None else len(self.data)
        else:
            assert size > 0
            self.size = size

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        x, y = zip(*elements)
        nb_eqs = [seq.count(self.env.func_separator) for seq in x]
        x = [torch.LongTensor([self.env.word2id[w] for w in seq]) for seq in x]
        y = [torch.LongTensor([self.env.word2id[w] for w in seq]) for seq in y]
        x, x_len = self.env.batch_sequences(x)
        y, y_len = self.env.batch_sequences(y)
        return (x, x_len), (y, y_len), torch.LongTensor(nb_eqs)

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if hasattr(self.env, "rng"):
            return
        if self.train:
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            self.env.rng = np.random.RandomState(
                [worker_id, self.global_rank, self.env_base_seed]
            )
            logger.info(
                f"Initialized random generator for worker {worker_id}, with seed "
                f"{[worker_id, self.global_rank, self.env_base_seed]} "
                f"(base seed={self.env_base_seed})."
            )
        else:
            self.env.rng = np.random.RandomState(0)

    def get_worker_id(self):
        """
        Get worker ID.
        """
        if not self.train:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0)
        return 0 if worker_info is None else worker_info.id

    def __len__(self):
        """
        Return dataset size.
        """
        return self.size

    def __getitem__(self, index):
        """
        Return a training sample.
        Either generate it, or read it from file.
        """
        self.init_rng()
        if self.path is None:
            return self.generate_sample()
        else:
            return self.read_sample(index)

    def read_sample(self, index):
        """
        Read a sample.
        """
        if self.train:
            index = self.env.rng.randint(len(self.data))
        x, y = self.data[index]
        x = x.split()
        y = y.split()
        assert len(x) >= 1 and len(y) >= 1
        return x, y

    def generate_sample(self):
        """
        Generate a sample.
        """
        while True:
            try:
                if self.task == "ode_convergence_speed":
                    xy = self.env.gen_ode_system_convergence()
                elif self.task == "ode_control":
                    if self.env.tau == 0:
                        xy = self.env.gen_control()
                    else:
                        xy = self.env.gen_control_t()
                elif self.task == "fourier_cond_init":
                    xy = self.env.gen_fourier_cond_init()
                else:
                    raise Exception(f"Unknown data type: {self.task}")
                if xy is None:
                    continue
                x, y = xy
                break
            except TimeoutError:
                continue
            except Exception as e:
                logger.error(
                    "An unknown exception of type {0} occurred for worker {4} "
                    'in line {1} for expression "{2}". Arguments:{3!r}.'.format(
                        type(e).__name__,
                        sys.exc_info()[-1].tb_lineno,
                        "F",
                        e.args,
                        self.get_worker_id(),
                    )
                )
                continue
        self.count += 1

        # clear SymPy cache periodically
        if CLEAR_SYMPY_CACHE_FREQ > 0 and self.count % CLEAR_SYMPY_CACHE_FREQ == 0:
            logger.warning(f"Clearing SymPy cache (worker {self.get_worker_id()})")
            clear_cache()

        return x, y
