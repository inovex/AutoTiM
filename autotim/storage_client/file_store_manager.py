from abc import ABC, abstractmethod


class FileStoreManager(ABC):

    def __init__(self):
        if self.__class__ == FileStoreManager:
            raise Exception('Oops! You have tried to instantiate an abstract class!')

    @abstractmethod
    def dir_exists(self, dir_path: str):
        raise NotImplementedError("You have called an abstract method!")

    @abstractmethod
    def download_dir(self, output_path, prefix: str = ""):
        raise NotImplementedError("You have called an abstract method!")

    @abstractmethod
    def save_single_file(self, src: str, dest: str):
        raise NotImplementedError("You have called an abstract method!")

class StorageDoesNotExistError(Exception):
    pass

class UploadToStorageFailedError(Exception):
    def __init__(self, src: str):
        self.message = f"Saving the file '{src}' to storage failed."
        super().__init__(self.message)

class DownloadFromStorageFailedError(Exception):
    def __init__(self, volume_name: str, prefix: str):
        self.message = f"Download dir '{volume_name}{prefix}' from storage failed."
        super().__init__(self.message)
