from candidate_generator import CandidateGenerator
from type_enums import RefactoringOperatorType
from pathlib import Path


if __name__ == "__main__":
    enable_llm_usage = False

    for operator in RefactoringOperatorType:
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

        print()
