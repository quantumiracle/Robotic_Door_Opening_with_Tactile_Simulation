'''
Add robolite/robosuite into sys.path
Don't need to `pip install -e .`
'''

import os
import sys

sys.path.insert(1, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'robolite'))

import robosuite

__all__ = []
