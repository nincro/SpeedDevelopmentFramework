# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 18:47:48 2019

@author: Administrator
"""
        
import sys
import os
import urllib.request
import tarfile
import zipfile

class Downloader(object):
    def __init__(self, url=None, path_to_save=None):
        self.url = url
        self.path_to_save = path_to_save
        

    def report_download_progress(self, count, block_size, total_size):
        pct_complete = float(count * block_size) / total_size
        msg = "\r {0:.1%} already downloaded".format(pct_complete)
        sys.stdout.write(msg)
        sys.stdout.flush()
    
    
    def download_data_url(self):
        url = self.url
        filename = url.split('/')[-1]
        file_path = os.path.join(self.path_to_save, filename)
        
        if not os.path.exists(file_path):
            os.makedirs(self.path_to_save, exist_ok=True)

            print("Download %s to %s" % (url, file_path))
            file_path, _ = urllib.request.urlretrieve(
                url=url,
                filename=file_path,
                reporthook=self.report_download_progress)
    
            print("\nExtracting files")
            if file_path.endswith(".zip"):
                zipfile.ZipFile(file=file_path, mode="r").extractall(self.path_to_save)
            elif file_path.endswith((".tar.gz", ".tgz")):
                tarfile.open(name=file_path, mode="r:gz").extractall(self.path_to_save)
