"""A parser that can parse FinQA formula."""

from typing import Any, Callable


class FinQAParser:
    """Parser class."""

    # class attributes
    ops = [
        ("add", lambda x, y: x + y),
        ("subtract", lambda x, y: x - y),
        ("multiply", lambda x, y: x * y),
        ("divide", lambda x, y: x / y),
        ("exp", lambda x, y: x**y),
        ("greater", lambda x, y: x > y),
    ]
    t_ops: list[tuple[str, Callable[[Any], Any]]] = [
        ("table_sum", sum),
        ("table_average", lambda x: sum(x) / len(x)),
        ("table_max", max),
        ("table_min", min),
    ]

    def __init__(self, table: str) -> None:
        """Initialize and compute the program."""
        self.table_str = table
        self.table_list = FinQAParser.parse_table(table)

    def parse(self, program_solution: str) -> str:
        """Parse a program to give result."""
        self.program_solution = program_solution
        ans, end = self.parse_prog(0)
        ans_str = ("yes" if ans else "no") if isinstance(ans, bool) else str(ans)
        if end != len(self.program_solution):
            raise ValueError("extra string at end")
        return ans_str

    @classmethod
    def parse_table(cls, table: str) -> list[list[str]]:
        """Parse a table to list of rows."""
        return [[x.strip() for x in row.split("|")[2:-1]] for row in table.splitlines()[2:]]

    @classmethod
    def parse_entry(cls, entry: str) -> float | None:
        """Parse an entry to a number. Return None means it's not a number."""
        if len(entry) == 0:
            return None
        if entry[0] == "$":
            if len(entry) < 2:
                raise ValueError("wrong format: $")
            entry = entry[2:]
        if len(entry) == 0:
            return None
        if "(" in entry:
            entry = entry.split("(")[0].strip()
        try:
            res = float(entry.replace("%", "e-2"))
        except ValueError:
            res = None
        return res

    def read_table(self, row_name: str) -> list[float]:
        """Read one row of numbers in table."""
        for row in self.table_list:
            if row[0] != row_name:
                continue
            # found row
            res = [FinQAParser.parse_entry(x) for x in row[1:]]
            if any(x is None for x in res):
                raise ValueError("cannot apply table operation with non-number entry.")
            return res  # type: ignore
        raise ValueError(f"cannot find row {row_name}.")

    def parse_prog(self, beg: int = 0) -> tuple[float | bool, int]:
        """Parse a program recursively from index `beg`."""
        prog = self.program_solution
        for op, func in FinQAParser.ops:
            op += "("
            if not prog.startswith(op, beg):
                continue
            beg += len(op)
            num1, beg = self.parse_prog(beg)
            if type(num1) is not float:
                raise TypeError("num1 not float")
            if not prog.startswith(", ", beg):
                raise ValueError("missing ,")
            num2, beg = self.parse_prog(beg + 2)
            if type(num2) is not float:
                raise TypeError("num2 not float")
            if not prog.startswith(")", beg):
                raise ValueError("missing )")
            return func(num1, num2), beg + 1

        for t_op, t_func in FinQAParser.t_ops:
            t_op += "("
            if not prog.startswith(t_op, beg):
                continue
            beg += len(t_op)
            ind = prog.find(",", beg)
            name, beg = prog[beg:ind], ind
            if not prog.startswith(", none)", beg):
                raise ValueError("missing none")
            return t_func(self.read_table(name)), beg + 7

        # parse number
        if prog.startswith("const_", beg):
            beg += 6
        end = beg
        while end < len(prog):
            if prog[end] in [",", ")"]:
                break
            end += 1
        return float(prog[beg:end].replace("%", "e-2").replace("m", "-")), end
