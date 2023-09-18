import streamlit as st
from streamlit_webrtc import webrtc_streamer
from st_audiorec import st_audiorec

from io import BytesIO
import threading
import time
import os

from gtts import gTTS # text to speech
import assemblyai as aai # speech to text
from deepface import DeepFace # emotion recognition
from langchain.llms import Clarifai # GPT-4
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain

system_role = '''
You are a psychological counselor that takes in emotion 
tracking data, personal info, and user thoughts on mind.
You are expected to give a descriptive response explaining
what different emotion values mean for them (if given), and write relevant
suggestions on how to cope with their negative feelings (if any).

There may be 3 inputs at most, ignore if any value is empty/none.
'''

human_template = '''
1. Emotion Dictionary Data Monitored by AI from Webcam
- Duration Key represents how long inference has gone
- Emotion Key elaborates composition of each emotion attribute
- Each value represent the number of frames that AI classifies as that particular emotion
DATA: {emo_tracker}
2. Personal Info: {p_info}
3. Thoughts on Mind: {thoughts}
'''

llm_prompt = ChatPromptTemplate.from_messages([
    ("system", system_role),
    ("human", human_template),
])

llm_model = Clarifai(pat=st.secrets['ClarifaiToken'], user_id='openai', 
                   app_id='chat-completion', model_id='GPT-4')

llm_chain = LLMChain(llm=llm_model, prompt=llm_prompt, verbose=True)

aai.settings.api_key = st.secrets['AssemblyAIToken']
transcriber = aai.Transcriber()

lock = threading.Lock()
img_container = {"img": None}
if 'tracker' not in st.session_state:
    st.session_state['tracker'] = {'emotions': {}, 'duration': 0, 'stress': {}}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img
    
    return frame

st.set_page_config('EmoCope')
st.title('EmoCope')

with st.expander('Expand for Help!'):
    st.write('1. Press Start, and AI will monitor your emotions overtime')
    st.write('2. Click Stop if AI alert is correct')
    st.write('3. Go to Counsel Tab to seek for psycological advice')

monitor_tab, counsel_tab = st.tabs(['Monitoring', 'Counseling'])

with monitor_tab:
    stream = webrtc_streamer(key="stream", video_frame_callback=video_frame_callback,
                            media_stream_constraints={'video': True, 'audio': False},
                            rtc_configuration={
                                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                            })

    emotion_metrics = st.empty()
    pulse_metrics = st.empty()
    message = st.empty()
    start = True

    while stream.state.playing:
        with lock:
            img = img_container["img"]
        if img is not None:
            try:
                faces = DeepFace.analyze(img_path = img, actions = ['emotion'],
                                        detector_backend='ssd', silent=True)
                if len(faces) == 1:
                    if start:
                        start_time = time.time()
                        start = False

                    bbox = faces[0]['region'] # x, y, w, h
                    # cropping
                    # image processing
                    cur_emo = faces[0]['dominant_emotion']
                    attributes = faces[0]['emotion']

                    # live metric display
                    with emotion_metrics:
                        cols = st.columns(7)

                        for idx, (emo, score) in enumerate(attributes.items()):
                            value = f'{round(score)}%'
                            if emo == cur_emo:
                                cols[idx].metric(emo, value, 'Dominant')
                            else:
                                cols[idx].metric(emo, value)

                    tracker = st.session_state['tracker']

                    if cur_emo in tracker['emotions'].keys():
                        tracker['emotions'][cur_emo] += 1
                    else:
                        tracker['emotions'][cur_emo] = 1

                    # track stress, bpm, hrv

                    cur_time = time.time()
                    duration = round(cur_time - start_time)
                    st.session_state['tracker']['duration'] = duration

                    if duration > 5: # seconds
                        dom_emo = max(tracker, key=tracker.get)
                        if dom_emo in ['angry', 'fear', 'sad']:
                            dom_frames = tracker[dom_emo]
                            total_frames = sum(tracker.values())
                            stats = round(dom_frames / total_frames * 100)
                            if stats > 50:
                                message.error(f'''High Stress Levels Detected!
                                            {stats}% {dom_emo} in {duration} secs!''')
                            else:
                                message.warning('Low Stress Levels Detected!')
                        else:
                            message.success('No Stress Detected!')

                    else:
                        message.info('Analyzing Emotions..')

                else:
                    message.error('Multiple Faces Detected!')

            except ValueError:
                message.error('No Face Detected!')


with counsel_tab:
    tracker = st.session_state['tracker']
    if tracker['duration'] > 0:
        use = st.toggle('Use Emotion Monitoring Data?')
    else:
        use = False
        st.info('Emotion Monitoring Data not Available!')

    # Retrieve stress index data? 
    
    personalize = st.toggle('Personalize counseling?')
    if personalize:
        with st.expander('Personalization'):
            name = st.text_input('Name')
            age = st.number_input('Age', 1, 100)
            gender = st.radio('Gender', ['Male', 'Female'], horizontal=True)

        p_info = f'Name: {name}; Age: {age}; Gender: {gender}'

    tell = st.toggle("Tell what's on your mind?")
    if tell:
        with st.expander('Context'):
            mode = st.radio('Mode', ['Speak', 'Type'])
            if mode == 'Speak':
                audio_bytes = st_audiorec()
                if audio_bytes:
                    file_name = 'temp_transcript.wav'
                    with open(file_name, "wb") as f:
                        f.write(audio_bytes)
                    
                    context = transcriber.transcribe(file_name).text
                    st.write(context)
                    os.remove(file_name)

            else:
                context = st.text_area('Text to Analyze')

    if use or tell:
        counsel = st.button('Counsel')
        if counsel:
            response = llm_chain.run(emo_tracker=[tracker['emotions'] if use else None],
                        p_info=[p_info if personalize else None],
                        thoughts=[context if tell else None])
            st.write(response)

            speech_bytes = BytesIO()
            tts = gTTS(response)
            tts.write_to_fp(speech_bytes)
            st.audio(speech_bytes)
