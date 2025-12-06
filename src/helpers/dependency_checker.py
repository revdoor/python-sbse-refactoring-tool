"""
This module defines the DependencyChecker class,
which is used to check the dependency for the nodes between the surroundings.
"""
import ast
from src.helpers.store_load_visitor import StoreLoadVisitor


class DependencyChecker:
    @staticmethod
    def _get_target_info(parent_node, attr_name, idx, length):
        if not hasattr(parent_node, attr_name):
            return None

        attr = getattr(parent_node, attr_name)
        if len(attr) < idx + length:
            return None

        return {
            'attr': attr,
            'target_nodes': attr[idx: idx + length],
            'prev_nodes': attr[:idx],
            'next_nodes': attr[idx + length:]
        }

    @staticmethod
    def is_dependency_free(top_lvl_node, parent_node, attr_name, idx, length):
        # dependency free: the stored in target should not be used outside, after the target

        info = DependencyChecker._get_target_info(parent_node, attr_name, idx, length)
        if info is None:
            return False

        target_nodes = info['target_nodes']
        prev_nodes = info['prev_nodes']

        target_visitor = StoreLoadVisitor()
        for target_node in target_nodes:
            target_visitor.visit(target_node)

        target_store_ids = target_visitor.store_ids

        next_visitor = StoreLoadVisitor(prev_nodes + target_nodes)
        next_visitor.visit(top_lvl_node)

        next_load_ids = next_visitor.load_ids

        return target_store_ids.isdisjoint(next_load_ids)

    @staticmethod
    def is_dependency_free_with_return(top_lvl_node, parent_node, attr_name, idx, length):
        # dependency free: the stored in target should not be used outside, after the target
        # consider only when the last target node is Assign or AugAssign
        # and suppose that we extract the method with return statement

        info = DependencyChecker._get_target_info(parent_node, attr_name, idx, length)
        if info is None:
            return False

        target_nodes = info['target_nodes']
        prev_nodes = info['prev_nodes']

        last_node = target_nodes[-1]
        body_nodes = target_nodes[:-1]

        if not isinstance(last_node, ast.Assign) and not isinstance(last_node, ast.AugAssign):
            return False

        target_visitor = StoreLoadVisitor()
        for target_node in body_nodes:
            target_visitor.visit(target_node)

        target_store_ids = target_visitor.store_ids

        next_visitor = StoreLoadVisitor(prev_nodes + target_nodes)
        next_visitor.visit(top_lvl_node)

        next_load_ids = next_visitor.load_ids

        return target_store_ids.isdisjoint(next_load_ids)
