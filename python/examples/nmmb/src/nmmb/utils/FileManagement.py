import os
import shutil


def deleteFile(filePath):
    """
    Deletes the file specified by the filePath.
    :param filePath: File path to delete.
    :return: Boolean. True if success, False if failed.
    """
    try:
        os.remove(filePath)
    except:
        return False
    return True


def deleteFileOrFolder(fileOrFolder):
    """
    Fault tolerant method to erase a file or folder recursively Does not raise any exception when file is not erased.
    :param fileOrFolder: File or folder path to be deleted.
    """
    try:
        if fileOrFolder is not None:
            if os.path.isdir(fileOrFolder):
                shutil.rmtree(fileOrFolder)
            else:
                # It is a file, we can delete it automatically
                os.remove(fileOrFolder)
        else:
            # The file is not valid, skip
            pass
    except:
        # Skip
        pass


def copyFile(srcPath, targetPath):
    """
    Copies a file from @srcPath to @targetPath.
    :param srcPath: Source file path.
    :param targetPath: Destination path.
    :return: Boolean. True if success, False if failed.
    """
    try:
        shutil.copy(srcPath, targetPath)
    except:
        return False
    return True


def moveFile(srcPath, targetPath):
    """
    Moves a file from @srcPath to @targetPath.
    :param srcPath: Source file path.
    :param targetPath: Destination path.
    :return: Boolean. True if success, False if failed.
    """
    try:
        shutil.move(srcPath, targetPath)
    except:
        return False
    return True


def createDir(folderPath):
    """
    Creates a directory and all its parents for @folderPath.
    :param folderPath: Folder path to create.
    :return: Boolean. True if success, False if failed.
    """
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
        return True
    else:
        return False
