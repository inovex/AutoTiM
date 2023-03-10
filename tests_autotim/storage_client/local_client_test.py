"""Prediction tests. """
# pylint: disable=redefined-builtin
import os
import random
import shutil
import unittest

from autotim.storage_client.local_client import LocalStorageClient, \
    VolumeDoesNotExistError


def get_non_existent_directory_path(base_dir: str):
    # try to generate a random dir name 10 times
    for _ in range(10):
        dir_name = 'dir_' + str(random.randint(0, 102))
        dir_path_name = os.path.join(base_dir, dir_name)

        if not os.path.exists(dir_path_name):
            return dir_name
    return None


class LocalClientTest(unittest.TestCase):
    base_dir = "./tests_autotim/"
    client = LocalStorageClient(base_dir)

    def test_local_client_throws_vol_does_not_exist_correctly(self):
        vol_name = get_non_existent_directory_path(base_dir='')

        self.assertRaises(VolumeDoesNotExistError,
                          LocalStorageClient, vol_name)

    def test_dir_exists_returns_true(self):
        # create directory
        dir_name = 'possum_named_dir'
        dir_path_name = os.path.join(self.base_dir, dir_name)
        if not os.path.exists(dir_path_name):
            os.mkdir(dir_path_name)

        self.assertEqual(self.client.dir_exists(dir_name), True)

        # remove created directory
        os.rmdir(dir_path_name)

    def test_dir_exists_returns_false(self):
        dir_name = get_non_existent_directory_path(base_dir=self.base_dir)

        if dir_name is not None:
            self.assertEqual(self.client.dir_exists(dir_name), False)

    def test_save_single_file_works_correctly(self):
        file_name = 'test_file.txt'
        with open(file_name, "w+") as f:
            for i in range(5):
                f.write("This is a test %d\r\n" % (i + 1))

        client_path_name = os.path.join(self.base_dir, file_name)
        self.client.save_single_file(src="./" + file_name, dest=".")

        value = os.path.isfile(client_path_name)
        self.assertEqual(value, True)

        # clean-up
        os.remove(file_name)
        os.remove(client_path_name)

    def test_download_dir_works_correctly(self):
        # Copy one of the directories
        dest_dir = './data_test_download'
        src_dir_name = 'test_objects'
        dest_path = os.path.join(dest_dir, src_dir_name)
        self.client.download_dir(prefix=src_dir_name, output_path=dest_path)

        value = os.path.isdir(dest_path)
        self.assertEqual(value, True)

        shutil.rmtree(dest_dir)


if __name__ == "__main__":
    unittest.main()
