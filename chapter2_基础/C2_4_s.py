from soundBase import soundBase

sb = soundBase('a.wav')
y = sb.vowel_generate(16000)
sb.audiowrite(y, 16000)
sb.soundplot()
sb.audioplayer()
