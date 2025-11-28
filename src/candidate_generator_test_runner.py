from candidate_generator import CandidateGenerator


if __name__ == "__main__":
    file = 'dump_target_code/dump_target_code_im.py'

    cg = CandidateGenerator()

    with open(file, 'r') as f:
        source_code = f.read()
        candidates = cg.generate_inline_method_candidates(source_code)

        for candidate in candidates:
            print(candidate)
