import ast


class ControlFlowChecker:
    @staticmethod
    def has_return(stmts: list[ast.stmt]) -> bool:
        for stmt in stmts:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Return):
                    return True
        return False
