import os
import vlc
import multitasking
import random
from gtts import gTTS
from time import sleep

speechQueue = True

@multitasking.task
def talk(message,timer):

	global speechQueue
	
	while not(speechQueue):
		pass
		
	speechQueue = False	
	
	sleep(timer)

	name = "output.mp3"
	
	tts= gTTS(text=message, lang='en')
	tts.save(name)

	media = vlc.MediaPlayer(name)
	duration = abs(media.get_length())
	media.play()
	sleep(duration)
	os.remove(name)
	
	speechQueue = True
	
talk("hello",0)
