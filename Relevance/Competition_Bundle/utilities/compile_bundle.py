import os
import zipfile


def zip_competition_bundle():
    folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    bundle_path = os.path.join(folder_path, 'Relevance_Bundle.zip')

    exclude_files = {'README.md'}
    include_files = {'logo.png', 'competition.yaml'}
    exclude_folders = {'utilities'}
    exclude_patterns = {'.DS_Store', '__pycache__'}

    with zipfile.ZipFile(bundle_path, 'w') as zipf:
        for root, dirs, files in os.walk(folder_path):
            rel_path = os.path.relpath(root, folder_path)

            if rel_path == '.':
                for file in files:
                    if file in include_files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, folder_path)
                        zipf.write(file_path, arcname=arcname)
            elif all(exclude not in rel_path for exclude in exclude_folders):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file not in exclude_files and not any(pattern in file for pattern in exclude_patterns):
                        arcname = os.path.relpath(file_path, folder_path)
                        zipf.write(file_path, arcname=arcname)


if __name__ == "__main__":
    zip_competition_bundle()
