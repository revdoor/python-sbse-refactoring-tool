from nsga_runner import NSGARunner


if __name__ == "__main__":
    source_code_path = "./dump_target_code/dump_target_code_im.py"

    with open(source_code_path, 'r') as f:
        source_code = f.read()
        print("Loaded source code from", source_code_path)

    runner = NSGARunner.from_source_code(source_code)
    runner.run()
