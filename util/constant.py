fer2013_classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
ckplus_classes = ["anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"]

def __const(set, result):
   '''
   Convenience method to determine values for certain datasets.
   '''
   if set == "fer2013":
      if result == "offsets":
         return (20, 40)
      if result == "classes":
         return fer2013_classes
