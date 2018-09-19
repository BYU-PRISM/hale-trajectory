# -*- coding: utf-8 -*-

from pydoc import locate
import sys

sys.path.insert(0,'./Aircraft/')
Aircraft = locate('Aircraft.aquila_e216.Aircraft')
x = Aircraft()