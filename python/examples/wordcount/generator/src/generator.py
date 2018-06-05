#!/usr/bin/python
#
#  Copyright 2002-2018 Barcelona Supercomputing Center (www.bsc.es)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# -*- coding: utf-8 -*-

""" Words generator """

import sys
import os
from loremipsum import *


def main():
    num_files = int(sys.argv[1])
    size = int(sys.argv[2])*1024*1024  # MBytes to Bytes
    dataset_path = "dataset_{}f_{}mb".format(num_files, (size/1024)/1024)
    os.mkdir(dataset_path)

    for i in range(num_files):
        text_size = 0
        file_name = "file{}.txt".format(i)
        path = os.path.join(dataset_path, file_name)
        with open(path, 'w') as f:
            while text_size < size:
                paragraph = generate_paragraph()
                f.write(paragraph[2])
                text_size = os.path.getsize(path)
        print("{} generated: size {} mb".format(file_name, (float(text_size)/1024.0)/1024.0))


if __name__ == "__main__":
    main()
