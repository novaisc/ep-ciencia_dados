def create_directory(directory_path):
    """
    Create a directory if it does not exist.

    Args:
        directory_path (str): The path of the directory to create.
    """
    import os

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")