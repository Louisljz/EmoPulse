import streamlit as st
from streamlit_webrtc import webrtc_streamer
import threading
import time
from deepface import DeepFace

st.set_page_config('Face Emotion Monitoring')
st.title('Face Emotion Monitoring')

lock = threading.Lock()
img_container = {"img": None}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img
    
    return frame

stream = webrtc_streamer(key="stream", video_frame_callback=video_frame_callback,
                         media_stream_constraints={'video': True, 'audio': False})

metrics_holder = st.empty()
message = st.empty()
tracker = {}
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

                cur_emo = faces[0]['dominant_emotion']
                attributes = faces[0]['emotion']

                with metrics_holder:
                    cols = st.columns(7)

                    for idx, (emo, score) in enumerate(attributes.items()):
                        value = f'{round(score)}%'
                        if emo == cur_emo:
                            cols[idx].metric(emo, value, 'Dominant')
                        else:
                            cols[idx].metric(emo, value)

                if cur_emo in tracker.keys():
                    tracker[cur_emo] += 1
                else:
                    tracker[cur_emo] = 1

                cur_time = time.time()
                duration = round(cur_time - start_time)

                if duration > 5:
                    dom_emo = max(tracker, key=tracker.get)
                    if dom_emo in ['angry', 'fear', 'sad']:
                        dom_frames = tracker[dom_emo]
                        total_frames = sum(tracker.values())
                        stats = round(dom_frames / total_frames * 100)
                        if stats > 50:
                            message.error(f'''High Stress Levels Detected!
                                        {stats}% {dom_emo} in {duration} secs!
                                        ''')
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
