def compare_bin_files(file1_path, file2_path):
    with open(file1_path, 'rb') as file1, open(file2_path, 'rb') as file2:
        file1_bytes = file1.read()
        file2_bytes = file2.read()
        
        if file1_bytes == file2_bytes:
            print("The files are identical.")
        else:
            print("The files are different.")

compare_bin_files('cpp_input.bin', 'python_input.bin')
