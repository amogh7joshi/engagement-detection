#!/usr/bin/env python3
# -*- coding = utf-8 -*-

# Custom errors, currently not in use.

class ModuleDoesNotExistError(ImportError):
   def __init__(self, module, message = None):
      # Custom ImportError Exception
      if message: self.message = message
      else:
         self.message = f"The module {module} is not installed, install it by running: pip install {module}"

class InvalidImageError(TypeError):
   def __init__(self, message = "Invalid Image."):
      # Custom TypeError Exception for invalid images.
      self.message = message