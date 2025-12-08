from collections.abc import Sequence
from pathlib import Path
from candidate_generator import CandidateGenerator
from type_enums import RefactoringOperatorType
from refactoring_operator import (
    RefactoringOperator,
    InlineMethodOperator,
    DecomposeConditionalOperator,
    ReverseConditionalExpressionOperator,
    ConsolidateConditionalExpressionOperator,
    ReplaceNestedConditionalOperator,
    RenameMethodOperator,
    RemoveDuplicateMethodOperator
)


class TestData:
    @staticmethod
    def get_test_data_for_operator(operator: RefactoringOperatorType) -> Sequence[RefactoringOperator]:
        match operator:
            case RefactoringOperatorType.IM:
                return TestData.get_test_data_for_im()
            case RefactoringOperatorType.DC:
                return TestData.get_test_data_for_dc()
            case RefactoringOperatorType.CC:
                return TestData.get_test_data_for_cc()
            case RefactoringOperatorType.RC:
                return TestData.get_test_data_for_rc()
            case RefactoringOperatorType.RNC:
                return TestData.get_test_data_for_rnc()
            case RefactoringOperatorType.RDM:
                return TestData.get_test_data_for_rdm()
            case RefactoringOperatorType.RM:
                return TestData.get_test_data_for_rm()
            case _:
                return []

    @staticmethod
    def get_test_data_for_im():
        return [
            InlineMethodOperator(1),
            InlineMethodOperator(2),
            InlineMethodOperator(3)
        ]

    @staticmethod
    def get_test_data_for_dc():
        return [
            DecomposeConditionalOperator(1, '1'),
            DecomposeConditionalOperator(1, '2'),
            DecomposeConditionalOperator(1, '3')
        ]

    @staticmethod
    def get_test_data_for_cc():
        return [
            ConsolidateConditionalExpressionOperator(1, 4),
            ConsolidateConditionalExpressionOperator(2, 3),
            ConsolidateConditionalExpressionOperator(3, 2)
        ]

    @staticmethod
    def get_test_data_for_rc():
        return [
            ReverseConditionalExpressionOperator(1),
            ReverseConditionalExpressionOperator(2),
            ReverseConditionalExpressionOperator(3),
            ReverseConditionalExpressionOperator(4),
            ReverseConditionalExpressionOperator(5),
            ReverseConditionalExpressionOperator(6),
            ReverseConditionalExpressionOperator(7)
        ]

    @staticmethod
    def get_test_data_for_rnc():
        return [
            ReplaceNestedConditionalOperator(1, 3),
            ReplaceNestedConditionalOperator(2, 2),
            ReplaceNestedConditionalOperator(7, 2),
            ReplaceNestedConditionalOperator(4, 2)
        ]

    @staticmethod
    def get_test_data_for_rdm():
        return [
            RemoveDuplicateMethodOperator(2, 1),
            RemoveDuplicateMethodOperator(3, 1),
            RemoveDuplicateMethodOperator(3, 2),
            RemoveDuplicateMethodOperator(5, 4)
        ]

    @staticmethod
    def get_test_data_for_rm():
        return [
            RenameMethodOperator(1, '1'),
            RenameMethodOperator(1, '2'),
            RenameMethodOperator(1, '3'),
            RenameMethodOperator(2, '1'),
            RenameMethodOperator(2, '2'),
            RenameMethodOperator(2, '3'),
            RenameMethodOperator(3, '1'),
            RenameMethodOperator(3, '2'),
            RenameMethodOperator(3, '3')
        ]


def compare_with_test_data(operator, result):
    if not operator.is_implemented():
        print(f"!!!Operator {operator.value} is not implemented. Skipping comparison!!!")
        return

    raw_test_data = TestData.get_test_data_for_operator(operator)
    test_data = [[td, False] for td in raw_test_data]

    # ignore the order
    for res in result:
        found = False

        for i in range(len(test_data)):
            if not test_data[i][1] and res == test_data[i][0]:
                test_data[i][1] = True
                found = True
                break

        if not found:
            print(f"Unexpected candidate: {res}")

    all_matched = all(td[1] for td in test_data)
    if not all_matched:
        for td, matched in test_data:
            if not matched:
                print(f"Missing candidate: {td}")


if __name__ == "__main__":
    enable_llm_usage = True

    target = [RefactoringOperatorType.EMR]

    for operator in target:
        print(f"Run for {operator.value}...")

        if not enable_llm_usage and operator.uses_llm():
            print(f"!!!LLM usage is disabled. Skipping {operator.value}!!!")
            print()
            continue

        script_dir = Path(__file__).parent.resolve()
        file_path = script_dir / f'dump_target_code/dump_target_code_{operator.name.lower()}.py'

        if not file_path.exists():
            print(f"!!!Dump code for {operator.value} does not exist. Skipping!!!")
            print()
            continue

        with open(file_path, 'r') as f:
            source_code = f.read()
            candidates = CandidateGenerator.generate_candidates_by_operator(source_code, operator)

            for candidate in candidates:
                print(candidate)

            compare_with_test_data(operator, candidates)

        print()
