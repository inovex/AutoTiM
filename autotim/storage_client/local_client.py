import logging
import os
from pathlib import Path
import shutil
from injector import inject

from autotim.storage_client.file_store_manager import FileStoreManager, \
    StorageDoesNotExistError, UploadToStorageFailedError, \
    DownloadFromStorageFailedError


class LocalStorageClient(FileStoreManager):
    """Client to store files (datasets) in a specified mounted volume.

    Child Class of FileStoreManager
    """
    @inject
    def __init__(self, directory_path = "/mnt/persistent-disk/data/"):
        FileStoreManager.__init__(self=self)
        self._directory_path = directory_path
        if not os.path.isdir(self._directory_path):
            raise VolumeDoesNotExistError(volume_name=self._directory_path)

    def dir_exists(self, dir_path: str):
        if not os.path.isdir(self._directory_path + dir_path):
            logging.debug("Directory does not exist")
            return False
        return True

    def save_single_file(self, src: str, dest: str):
        try:
            path = os.path.join(self._directory_path, dest)
            Path(path).mkdir(parents=True, exist_ok=True)
            shutil.copy(src, path)
        except shutil.SameFileError:
            logging.info('Source and destination represents the same file.')
        except (FileNotFoundError, OSError, IOError, PermissionError) as e:
            if isinstance(e, PermissionError):
                logging.error("Permission denied while uploading file with LocalStorageClient.")
            raise UploadToStorageFailedError(src) from e

    def download_dir(self, output_path, prefix: str = ""):
        try:
            # check if a tree / part of the tree to be copied exists
                # if so -> delete, otherwise shutil.copytree fails
            copy_dest = os.path.join(output_path, prefix)
            if os.path.exists(copy_dest):
                shutil.rmtree(copy_dest)

            copy_src = os.path.join(self._directory_path, prefix)
            shutil.copytree(copy_src, copy_dest)
        except (FileNotFoundError, OSError, IOError, PermissionError) as e:
            if isinstance(e, PermissionError):
                logging.warning("Permission denied while downloading file with LocalStorageClient.")
            raise DownloadFromStorageFailedError(
                volume_name=self._directory_path, prefix=prefix) from e


class VolumeDoesNotExistError(StorageDoesNotExistError):
    def __init__(self, volume_name: str):
        self.message = f"Docker Volume '{volume_name}' does not exist.\n\
            Local storage cannot be used."
        super().__init__(self.message)
