from candidate_generator import CandidateGenerator
from type_enums import RefactoringOperatorType


if __name__ == "__main__":
    for operator in RefactoringOperatorType:
        print(f"Run for {operator.value}...")
        file = f'dump_target_code/dump_target_code_{operator.name.lower()}.py'

        with open(file, 'r') as f:
            source_code = f.read()
            candidates = CandidateGenerator.generate_candidates_by_operator(source_code, operator)

            for candidate in candidates:
                print(candidate)

        print()
