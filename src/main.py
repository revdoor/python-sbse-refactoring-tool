from nsga_runner import NSGARunner
import sys
import os


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    source_code_path = os.path.join(project_root, 'src', 'dump_target_code', 'dump_target_code_im.py')
    # source_code_path = "./dump_target_code/dump_target_code_im.py"

    with open(source_code_path, 'r') as f:
        source_code = f.read()
        print("Loaded source code from", source_code_path)

    runner = NSGARunner.from_source_code(source_code)
    runner.run()
