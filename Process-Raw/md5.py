import hashlib
import binascii


def md5sum(path, blocksize=65536):
    """
    Generates the md5 of a file.
    (Inspired from http://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file)
    :param path:
    :param blocksize:
    :return:
    """
    hasher = hashlib.md5()
    with open(path, 'rb') as afile:
        buf = afile.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(blocksize)
        return binascii.hexlify(hasher.digest())


def check_md5(path_file, path_md5_file):
    with open(path_md5_file, 'r') as f:
        given_md5 = f.readline().strip()
    computed_md5 = md5sum(path_file)
    return given_md5 == computed_md5.decode()