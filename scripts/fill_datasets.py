"""
Fill all dataset JSON files with 30+ realistic examples.
Run this once to ensure every intent has enough training data.
"""
import json
import pathlib

DATASETS_DIR = pathlib.Path("datasets/intents")

# Generate 30+ examples for every dataset file that has <30
data = {
    'brightness_up.json': [
        'increase brightness', 'brightness up', 'brighter', 'more brightness',
        'turn up brightness', 'raise brightness', 'make it brighter',
        'increase screen brightness', 'brighten the screen', 'brighten display',
        'more light', 'make screen brighter', 'raise screen brightness',
        'turn brightness up', 'increase display brightness', 'brightness increase',
        'need more brightness', 'screen too dark', 'brighten up', 'boost brightness',
        'max brightness', 'full brightness', 'higher brightness', 'set brightness high',
        'make the screen bright', 'brighten my screen', 'increase the brightness',
        'want it brighter', 'too dim', 'crank up the brightness', 'boost the light',
        'brightness max', 'brighten to max', 'full screen brightness'
    ],
    'brightness_down.json': [
        'decrease brightness', 'brightness down', 'dimmer', 'less brightness',
        'turn down brightness', 'lower brightness', 'reduce brightness',
        'decrease screen brightness', 'dim the screen', 'dim display',
        'less light', 'make screen dimmer', 'lower screen brightness',
        'turn brightness down', 'decrease display brightness', 'brightness decrease',
        'need less brightness', 'screen too bright', 'dim it down', 'reduce light',
        'min brightness', 'zero brightness', 'lower brightness level',
        'set brightness low', 'dim my screen', 'want it dimmer', 'too bright',
        'crank down the brightness', 'softer light', 'lower the brightness',
        'brightness min', 'dim to minimum', 'lowest brightness'
    ],
    'date_query.json': [
        'what is the date', 'what date is it', 'tell me the date',
        'what day is it', 'current date', 'todays date',
        'whats the date today', 'date please', 'give me the date',
        'show me todays date', 'what day is today', 'tell me what day it is',
        'todays day', 'day of week', 'what is today', 'current day',
        'whats today', 'date today', 'show date', 'display date',
        'which day', 'whats the date today', 'tell me todays date',
        'what day of the week', 'can you tell me the date', 'whats the day',
        'get the date', 'check date', 'what month is it', 'current month',
        'whats the day today', 'date of today'
    ],
    'exit.json': [
        'goodbye', 'bye', 'exit', 'quit', 'see you later', 'good night',
        'talk to you later', 'later', 'see ya', 'take care', 'bye bye',
        'i am leaving', 'shutdown', 'power off', 'sleep', 'good bye',
        'see you soon', 'catch you later', 'have to go', 'leaving now',
        'close Diego', 'stop', 'end', 'finish', 'done', 'thats all',
        'im done', 'turn off', 'switch off', 'go to sleep', 'farewell',
        'im leaving now', 'thats enough', 'i am done'
    ],
    'greeting.json': [
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
        'whats up', 'yo', 'hey there', 'howdy', 'greetings', 'nice to see you',
        'hello there', 'hiya', 'hey buddy', 'morning', 'good day', 'hi Diego',
        'hello Diego', 'yo Diego', 'hey Diego', 'hi there', 'welcome', 'pleased to meet you',
        'how do you do', 'hi friend', 'hey friend', 'greetings Diego', 'hello again',
        'good to see you', 'long time no see', 'welcome back', 'hi hi'
    ],
    'help.json': [
        'what can you do', 'help', 'show commands', 'list features',
        'what are your capabilities', 'how can you help me', 'tell me what you can do',
        'available commands', 'help me', 'what do you do', 'show help',
        'help please', 'i need help', 'capabilities', 'show features',
        'list commands', 'what functions do you have', 'how to use you',
        'give me help', 'assist me', 'what are your skills', 'features list',
        'show me commands', 'display help', 'help options', 'tell me commands',
        'what commands are available', 'what can i ask', 'list all features',
        'show me what you can do', 'command list', 'show help menu'
    ],
    'joke.json': [
        'tell me a joke', 'make me laugh', 'say something funny',
        'tell a joke', 'crack a joke', 'do you know any jokes',
        'give me a joke', 'joke', 'tell me something funny',
        'make me smile', 'entertain me', 'say a joke', 'humor me',
        'another joke', 'got any jokes', 'joke please', 'funny joke',
        'tell a funny story', 'amuse me', 'crack me up',
        'make me happy', 'say something humorous', 'i want a joke',
        'give me a laugh', 'cheer me up', 'lighten the mood',
        'tell a humorous tale', 'i need a joke', 'make me chuckle',
        'give me a funny joke', 'tell me a funny story'
    ],
    'news.json': [
        'what is the news', 'latest news', 'news headlines',
        'tell me the news', 'current affairs', 'what is happening',
        'any breaking news', 'news', 'show me news', 'get news',
        'headlines', 'top stories', 'news update', 'daily news',
        'news today', 'breaking news', 'current events', 'world news',
        'local news', 'tech news', 'sports news', 'whats new',
        'give me the news', 'news report', 'update me', 'latest headlines',
        'news summary', 'tell me what happened', 'news now', 'hot topics',
        'top news', 'news headlines today'
    ],
    'telegram_read.json': [
        'read telegram', 'check telegram messages', 'read my messages',
        'check telegram', 'read latest message', 'show my telegrams',
        'whats new on telegram', 'telegram messages', 'read telegrams',
        'check my telegrams', 'show telegrams', 'get telegram messages',
        'read my telegram', 'telegram updates', 'show my messages',
        'read new messages', 'check my messages', 'telegram inbox',
        'view telegrams', 'display telegrams', 'telegram read',
        'read my chats', 'check chats', 'open telegram messages',
        'see my telegrams', 'latest telegram', 'unread messages',
        'show unread', 'read my chats', 'whats on telegram',
        'check new messages', 'open my telegram'
    ],
    'telegram_reply.json': [
        'reply on telegram', 'reply to message', 'reply back',
        'respond on telegram', 'send a reply', 'reply to telegram',
        'answer on telegram', 'reply message', 'reply that',
        'respond back', 'send reply', 'telegram reply',
        'answer message', 'reply to that', 'say in reply',
        'telegram response', 'reply to chat', 'answer back',
        'reply to them', 'respond to message', 'write reply',
        'compose reply', 'reply to telegram message', 'reply to it',
        'answer on telegram', 'send a response', 'reply quickly',
        'reply now', 'respond to that', 'answer that message',
        'write back', 'respond please'
    ],
    'telegram_send.json': [
        'send a telegram', 'send message on telegram',
        'message someone on telegram', 'send telegram to',
        'text on telegram', 'send a message via telegram',
        'send a text on telegram', 'telegram send', 'send message',
        'send telegram message', 'send via telegram', 'text telegram',
        'send a telegram message', 'message on telegram',
        'send text on telegram', 'send telegram text',
        'send a quick message', 'write on telegram', 'send a note',
        'send message to contact', 'send chat', 'telegram message',
        'send via telegram', 'send it on telegram', 'send a telegram now',
        'message a contact', 'send a telegram fast', 'compose telegram',
        'telegram quick message', 'send to telegram contact',
        'telegram someone', 'telegram a friend'
    ],
    'time_query.json': [
        'what time is it', 'tell me the time', 'current time',
        'whats the time', 'show me the time', 'time now',
        'give me the time', 'time please', 'what time',
        'show time', 'display time', 'check time',
        'what is the time', 'can you tell the time', 'time check',
        'get time', 'current time please', 'whats the time now',
        'time of day', 'tell time', 'may i know the time',
        'clock time', 'give time', 'the time please', 'what time is it now',
        'ask for time', 'know the time', 'time right now', 'exact time',
        'current local time', 'tell me the current time'
    ],
    'volume_down.json': [
        'decrease volume', 'volume down', 'turn down volume',
        'quieter', 'lower volume', 'reduce volume', 'softer',
        'volume decrease', 'lower the volume', 'turn it down',
        'make it quieter', 'reduce sound', 'lower sound', 'down volume',
        'less volume', 'decrease sound', 'turn volume down',
        'quieter please', 'less sound', 'turn it down a notch',
        'make it softer', 'cut the volume', 'mute volume',
        'volume low', 'set volume low', 'bring volume down',
        'lower the sound', 'reduce the volume', 'turn down the sound',
        'volume lower', 'make the sound lower'
    ],
    'volume_up.json': [
        'increase volume', 'volume up', 'turn up volume',
        'louder', 'raise volume', 'higher volume', 'make it louder',
        'volume increase', 'raise the volume', 'turn it up',
        'increase sound', 'raise sound', 'up volume', 'more volume',
        'louder please', 'more sound', 'turn volume up',
        'increase the volume', 'make it louder please', 'crank it up',
        'max volume', 'full volume', 'higher sound',
        'set volume high', 'bring volume up', 'raise the sound',
        'increase the sound', 'turn up the sound',
        'volume higher', 'make the sound louder'
    ],
    'weather_query.json': [
        'whats the weather', 'weather forecast', 'hows the weather',
        'whats the temperature', 'is it raining', 'weather outside',
        'current weather', 'tell me the weather', 'weather',
        'weather report', 'forecast', 'weather today',
        'what is the temperature', 'how is the weather', 'weather update',
        'weather condition', 'outside temp', 'whats it like outside',
        'should i take an umbrella', 'is it going to rain',
        'how hot is it', 'how cold is it', 'temperature outside',
        'weather check', 'check weather', 'tell me the temperature',
        'whats the forecast', 'will it rain', 'weather info',
        'weather for today', 'current temperature'
    ],
    'who_am_i.json': [
        'who am i', 'whats my name', 'tell me my name',
        'do you know me', 'who is the user', 'who i am',
        'identify me', 'what is my identity', 'who am i using this',
        'my name', 'whats my identity', 'tell me about me',
        'who uses this', 'recognize me', 'who is this',
        'am i known', 'my profile', 'my info', 'who i be',
        'what do you know about me', 'do you remember me',
        'my username', 'who is speaking', 'whats my user name',
        'tell me who i am', 'who is the person', 'identify yourself to me',
        'who is the user of this', 'what is my name', 'my user info'
    ],
    'youtube_close.json': [
        'close youtube', 'exit youtube', 'stop youtube',
        'close the video', 'end youtube', 'quit youtube',
        'close youtube player', 'stop playing video', 'exit video',
        'close the youtube', 'shut down youtube', 'youtube exit',
        'close the player', 'stop the video', 'end the video',
        'go back from youtube', 'leave youtube', 'stop the player',
        'close youtube tab', 'close the youtube video',
        'stop the music', 'end the stream', 'close the stream',
        'youtube stop', 'exit the video', 'close video player',
        'close youtube app', 'stop the youtube video', 'quit the video',
        'end playback', 'stop the playback'
    ],
    'youtube_previous.json': [
        'previous video', 'go back', 'previous track',
        'play the previous one', 'go back a video', 'last video',
        'previous song', 'go to previous', 'play previous',
        'go back one', 'previous in playlist', 'back to last',
        'previous item', 'go to last', 'play last video',
        'previous music', 'back to previous', 'play the last one',
        'previous in queue', 'go back to previous', 'previous track please',
        'last song', 'play previous song', 'go to previous video',
        'back to the last video', 'play the previous track',
        'previous video please', 'go to the last', 'play the last song',
        'previous in line', 'previous in the list'
    ],
    'youtube_seek_backward.json': [
        'rewind', 'go back', 'skip backward', 'jump back',
        'rewind 10 seconds', 'go backward', 'back up',
        'rewind a bit', 'go back some', 'skip back',
        'take it back', 'rewind the video', 'go back in video',
        'backward', 'go back a few seconds', 'reverse',
        'go backwards', 'rewind the song', 'go back a bit',
        'rewind 30 seconds', 'rewind 1 minute', 'take me back',
        'roll back', 'go to earlier', 'back it up', 'play earlier',
        'backward skip', 'go back a little', 'rewind a minute'
    ],
    'youtube_seek_forward.json': [
        'skip forward', 'fast forward', 'jump forward',
        'skip ahead', 'go forward', 'forward 10 seconds',
        'fast forward a bit', 'skip the ad', 'go ahead',
        'skip this part', 'move forward', 'advance',
        'go to later part', 'fast forward 30 seconds',
        'skip 1 minute', 'jump ahead', 'go to next part',
        'move ahead', 'forward skip', 'skip the song part',
        'go further', 'go to next section', 'skip to later',
        'fast forward the video', 'go forward in video',
        'skip forward 10', 'fast forward a bit', 'jump ahead a bit'
    ],
    'youtube_speed_down.json': [
        'decrease speed', 'slow down', 'slower', 'play slower',
        'decrease playback speed', 'go slower', 'reduce speed',
        'slow the video', 'play at slower speed', 'slower playback',
        'reduce playback speed', 'lower speed', 'make it slower',
        'slow down the video', 'play slower please',
        'decrease the speed', 'play at lower speed', 'slow video',
        'reduce the speed', 'set speed lower', 'speed decrease',
        'play back slower', 'slow motion', 'half speed', 'quarter speed',
        'go at half speed', 'reduce the playback speed', 'set to slower'
    ],
    'youtube_speed_up.json': [
        'increase speed', 'speed up', 'faster', 'play faster',
        'increase playback speed', 'go faster', 'speed it up',
        'faster playback', 'play at higher speed', 'speed increase',
        'increase the speed', 'play at faster speed', 'fast video',
        'make it faster', 'speed up the video', 'play faster please',
        'boost speed', 'faster speed', 'set speed higher',
        'speed up playback', 'play back faster', 'double speed',
        'go at double speed', 'increase the playback speed', 'set to faster'
    ],
    'youtube_unmute.json': [
        'unmute', 'unmute the video', 'turn on sound',
        'restore sound', 'enable audio', 'unmute audio',
        'turn sound on', 'enable sound', 'unmute the audio',
        'restore audio', 'sound on', 'audio on',
        'unmute video', 'turn on volume', 'enable volume',
        'unmute the music', 'bring back sound', 'audio back',
        'sound back', 'turn audio on', 'let me hear it',
        'unmute the song', 'restore the audio', 'enable the sound',
        'unmute the player', 'unmute playback', 'turn the sound on'
    ],
    'youtube_volume_down.json': [
        'lower youtube volume', 'youtube quieter', 'youtube volume down',
        'decrease youtube volume', 'reduce youtube volume',
        'youtube sound down', 'lower video volume', 'youtube volume decrease',
        'make youtube quieter', 'turn down youtube', 'youtube less loud',
        'reduce video volume', 'youtube sound lower', 'youtube down',
        'lower the music volume', 'decrease song volume', 'youtube quieter please',
        'youtube lower', 'quiet on youtube', 'youtube less volume',
        'reduce youtube sound', 'turn down the video', 'lower the video sound'
    ],
    'youtube_volume_up.json': [
        'increase youtube volume', 'youtube louder', 'youtube volume up',
        'louder on youtube', 'raise youtube volume', 'youtube sound up',
        'increase video volume', 'youtube volume increase',
        'make youtube louder', 'turn up youtube', 'youtube more loud',
        'raise video volume', 'youtube sound higher', 'youtube up',
        'raise the music volume', 'increase song volume', 'youtube louder please',
        'youtube higher', 'loud on youtube', 'youtube more volume',
        'increase youtube sound', 'turn up the video', 'raise the video sound'
    ],
}

for fname, examples in data.items():
    intent_name = fname.split('.')[0]
    examples = examples[:35]
    filepath = DATASETS_DIR / fname
    with open(filepath, 'w') as f:
        json.dump({"intent": intent_name, "description": "", "examples": examples}, f, indent=2)
        f.write('\n')
    print(f'{fname}: {len(examples)} examples')

print()
print("Verifying all datasets have 30+ examples...")
all_ok = True
for fpath in sorted(DATASETS_DIR.glob('*.json')):
    with open(fpath) as f:
        d = json.load(f)
    if isinstance(d, dict) and 'examples' in d:
        n = len(d['examples'])
    elif isinstance(d, list):
        n = len(d)
    else:
        n = 0
    status = "OK" if n >= 30 else "FAIL"
    if n < 30:
        all_ok = False
    print(f'  {status}: {fpath.name}: {n} examples')

if all_ok:
    print("ALL DATASETS HAVE 30+ EXAMPLES")
else:
    print("SOME DATASETS STILL BELOW 30")