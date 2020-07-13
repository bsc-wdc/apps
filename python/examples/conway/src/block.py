#!/usr/bin/python
#
#  Copyright 2002-2020 Barcelona Supercomputing Center (www.bsc.es)
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

# For better print formatting
from __future__ import print_function


class Block(object):

    def __init__(self, block_size=None, ref=None, rand=0):
        from random import randint
        if ref is None:
            if rand == 1:
                self._blockSize = block_size
                self._matrix = []
                for i in range(0, self._blockSize):
                    a = []
                    for j in range(0, self._blockSize):
                        a.append(randint(0, 1))
                    self._matrix.append(a)
            else:
                self._blockSize = block_size
                self._matrix = []
                for i in range(0, self._blockSize):
                    a = []
                    for j in range(0, self._blockSize):
                        a.append(0)
                    self._matrix.append(a)
        else:
            self._blockSize = ref._blockSize
            self._matrix = []
            for i in range(0, self._blockSize):
                a = []
                for j in range(0, self._blockSize):
                    a.append(ref._matrix[i][j])
                self._matrix.append(a)

    def set(self, i, j, val):
        self._matrix[i][j] = val

    def get(self, i, j):
        return self._matrix[i][j]
