#
# Author: Daniel Dittenhafer
#
#     Created: Sept 27, 2016
#
# Description: Main entry point for emotional faces data set creator.
#
__author__ = 'Daniel Dittenhafer'

import os

def list_files(path, recurse):
    # returns a list of names (with extension, without full path) of all files
    # in folder path
    files = []
    for name in os.listdir(path):
        filepath = os.path.join(path, name)
        if os.path.isfile(filepath):
            files.append(filepath)
        elif recurse:
            files += list_files(filepath, recurse)

    return files

def main():
    """Our main function."""
    path = "C:\Users\Dan\Downloads\lfw-deepfunneled"

    theFiles = list_files(path, True)
    print(len(theFiles))



# This is the main of the program.
if __name__ == "__main__":
    main()