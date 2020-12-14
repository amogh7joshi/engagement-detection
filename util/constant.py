fer2013_classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
ckplus_classes = ["anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"]
modified_emotion_classes = ['neutral', 'surprise', 'emotional', 'expressive']

def __const(set, result):
   '''
   Convenience method to determine values for certain datasets.
   '''
   if set == "fer2013":
      if result == "offsets":
         return (20, 40)
      if result == "classes":
         return fer2013_classes

# Personal Convenience
CENTER_X = 100
CENTER_Y = 80
CENTER_POS = (CENTER_X, CENTER_Y)
