from applier import Applier
from candidate_generator import CandidateGenerator
from type_enums import RefactoringOperatorType
from pathlib import Path
from refactoring_operator import RenameFieldOperator


if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    target_dir = script_dir / 'dump_target_code'

    applier = Applier()

    test_operators = [
        # RefactoringOperatorType.RC,
        # RefactoringOperatorType.CC,
        # RefactoringOperatorType.RNC,
        # RefactoringOperatorType.RM,
        # RefactoringOperatorType.IM,
        # RefactoringOperatorType.RDM,
        # RefactoringOperatorType.EM,
        # RefactoringOperatorType.EMR,
        # RefactoringOperatorType.RF,
        RefactoringOperatorType.DC,
    ]

    for operator in test_operators:
        file_path = target_dir / f'dump_target_code_{operator.name.lower()}.py'
        
        if not file_path.exists():
            continue

        print(f"\n\n[{operator.name}] Processing file: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            original_source_code = f.read()

        candidates = CandidateGenerator.generate_candidates_by_operator(original_source_code, operator)

        if not candidates:
            print(f">> No candidates found for {operator.name}.")
            continue

        print(f">> Found {len(candidates)} candidates.")

        for i, candidate in enumerate(candidates):
            if candidate.operator_type in (RefactoringOperatorType.RM, RefactoringOperatorType.EM, RefactoringOperatorType.EMR, RefactoringOperatorType.RF, RefactoringOperatorType.DC):
                if i % 3 != 0:
                    continue

            print(f"\n>> Applying Candidate {i+1}/{len(candidates)}: {candidate}")
            
            try:
                modified_code = applier.apply_refactoring(original_source_code, candidate)
                
                print(modified_code)
                
            except Exception as e:
                print(f"\033[91m!! Error applying candidate {i+1}: {e}\033[0m")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*20} Test Finished {'='*20}")