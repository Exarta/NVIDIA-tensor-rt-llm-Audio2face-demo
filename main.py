from fastapi import FastAPI, WebSocket ,  WebSocketDisconnect , Request
from fastapi.responses import HTMLResponse
import logging
import wave
import sys
import grpc
import time
import numpy as np
import soundfile
import audio2face_pb2
import audio2face_pb2_grpc
import riva.client
import riva.client.audio_io
import requests
# from tensor_llama_llm import get_query_response  , get_chat_response
from pydantic import BaseModel
import requests
from pydub import AudioSegment
# from ffmpeg import input, output
import os
from contextlib import asynccontextmanager
import pprint
from typing import List


#eleven labs vars

headers= {
  "Accept": "audio/wav",
  "Content-Type": "application/json",
  "xi-api-key": "cdadb7bd2efb978c726a897f96cadd1a"
}

# a2f componenets
instance = "/World/audio2face/CoreFullface"
StreamLiveLink =  "/World/audio2face/StreamLivelink"
BlendShapeSolver =  "/World/audio2face/BlendshapeSolve"
a2f_player_streaming = "/World/audio2face/audio_player_streaming"
a2f_player_regular = "/World/audio2face/Player"

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     #a2f_api_vars
#     file_path = r"D:\serve_unreal_sockets\mark_regular.usd"
#     a2f_current_audio_files_folder = "D:/serve_unreal_sockets/"
#     url_load = 'http://localhost:8011/A2F/USD/Load'
#     url_activatestreamlivelink = 'http://localhost:8011/A2F/Exporter/ActivateStreamLivelink'
#     url_a2e_streaming = 'http://localhost:8011/A2F/A2E/EnableStreaming'
#     url_a2e_autogen_onchange = 'http://localhost:8011/A2F/A2E/EnableAutoGenerateOnTrackChange'
#     url_set_track_loop = 'http://localhost:8011/A2F/Player/SetLooping'
#     url_get_current_track = 'http://localhost:8011/A2F/Player/GetCurrentTrack'
#     url_set_current_track = 'http://localhost:8011/A2F/Player/SetTrack'
#     url_get_root_path = 'http://localhost:8011/A2F/Player/GetRootPath'
#     url_set_root_path = 'http://localhost:8011/A2F/Player/SetRootPath'
#     url_setstreamlivelinksettings = 'http://localhost:8011/A2F/Exporter/SetStreamLivelinkSettings'
#     url_get_tracks = 'http://localhost:8011/A2F/Player/GetTracks'
    
#     #load a2f model
#     body_load_usd = {
#     'file_name': file_path
#     }
#     response_load = requests.post(url=url_load , json=body_load_usd)
#     pprint.pprint(response_load.json())
    
#     #set root path for audio files
#     body_root_path = {
#         "a2f_player": a2f_player_regular,
#         "dir_path" : a2f_current_audio_files_folder
#     }
#     response_root_path = requests.post(url=url_set_root_path , json=body_root_path)
#     pprint.pprint(f" root path set {response_root_path.json()}")
    
#     #get tracks in root path
#     body_get_tracks = {
#         "a2f_player": a2f_player_regular
#     }
#     response_tracks = requests.post(url=url_get_tracks , json=body_get_tracks)
#     pprint.pprint(f"tracks list {response_tracks.json()}")
    
    
#     #set track
#     body_set_track = {
#         "a2f_player": a2f_player_regular,
#         "file_name": 'output.wav',
#         "time_range": [
#              0,
#             -1
#         ]
#     }
    
#     response_set_track = requests.post(url=url_set_current_track , json=body_set_track)
#     pprint.pprint(response_set_track.json())
    
    
#     #set track loop to false
#     body_set_track_loop = {
#         "a2f_player": a2f_player_regular,
#         "loop_audio": False
#     }
#     response_tracks_loop = requests.post(url=url_set_track_loop , json=body_set_track_loop)
#     pprint.pprint(f"tracks list {response_tracks_loop.json()}")
    
    
#     #set live link settings
#     body_live_ink_settings = {
#     "node_path": StreamLiveLink,
#     "values": {"enable_audio_stream": True ,  "livelink_host": '172.16.15.216'  , "enable_gaze": False , "enable_idle_head": False }
#     }
#     response_live_link_settings = requests.post(url=url_setstreamlivelinksettings , json=body_live_ink_settings)
#     pprint.pprint(f" livelink setting {response_live_link_settings.json()}")
    
#     #enablae A2E auto gen on track change
#     body_a2e_auto_gen = {
#     "a2f_instance": instance ,
#     "enable": True 
#     }
#     response_a2e_auto_gen = requests.post(url=url_a2e_autogen_onchange , json=body_a2e_auto_gen)
#     pprint.pprint(F"enable A2E auro gen on change {response_a2e_auto_gen.json()}")
    
#     #enable A2E streaming
#     body_a2e_stream = {
#     "a2f_instance": instance ,
#     "enable": True 
#     }
#     response = requests.post(url=url_a2e_streaming , json=body_a2e_stream)
#     pprint.pprint(F"enable A2E Streaming {response.json()}")
    
    
#     #activate live link
#     body_activate_live_link = {
#     "node_path": StreamLiveLink ,
#     "value": True
#     }
#     response_activate_live_link = requests.post(url=url_activatestreamlivelink , json=body_activate_live_link)
#     pprint.pprint(f" Activate streamlive link {response_activate_live_link.json()}")
    
#     yield
        

app = FastAPI() # lifespan=lifespan
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s'
)
class Item(BaseModel):
    text: str

class StructData(BaseModel):

    string_field: str
    bool_field: bool
    vector_field: List[float] 

boosted_words = ["swift", "vision", "aqua", "gaze", "glasses", "mind", "lens"]
boosted_lm_score = 20.0
riva_uri = '172.16.15.209:50051'  # ip address of NVIDIA RIVA ASR and TTS

offline_output_file = 'test_audio.wav'
language_code = 'en-US'
sample_rate = 48000
nchannels = 1
sampwidth = 2




def push_audio_track(url, audio_data, samplerate, instance_name):
    """
    This function pushes the whole audio track at once via PushAudioRequest()
    PushAudioRequest parameters:
     * audio_data: bytes, containing audio data for the whole track, where each sample is encoded as 4 bytes (float32)
     * samplerate: sampling rate for the audio data
     * instance_name: prim path of the Audio2Face Streaming Audio Player on the stage, were to push the audio data
     * block_until_playback_is_finished: if True, the gRPC request will be blocked until the playback of the pushed track is finished
    The request is passed to PushAudio()
    """

    block_until_playback_is_finished = True  # ADJUST
    with grpc.insecure_channel(url) as channel:
        stub = audio2face_pb2_grpc.Audio2FaceStub(channel)
        request = audio2face_pb2.PushAudioRequest()
        request.audio_data = audio_data.astype(np.float32).tobytes()
        request.samplerate = samplerate
        request.instance_name = instance_name
        request.block_until_playback_is_finished = block_until_playback_is_finished
        print("Sending audio data...")
        response = stub.PushAudio(request)
        if response.success:
            print("SUCCESS")
        else:
            print(f"ERROR: {response.message}")
    print("Closed channel")



class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

def convertToAudioAndPlay(question, language):
    """
    This demo script shows how to send audio data to Audio2Face Streaming Audio Player via gRPC requests.
    There two options:
     * Send the whole track at once using PushAudioRequest()
     * Send the audio chunks seuqntially in a stream using PushAudioStreamRequest()
    For the second option this script emulates the stream of chunks, generated by splitting an input WAV audio file.
    But in a real application such stream of chunks may be aquired from some other streaming source:
     * streaming audio via internet, streaming Text-To-Speech, etc
    gRPC protocol details could be find in audio2face.proto
    """


    # Sleep time emulates long latency of the request
    sleep_time = 0.2  # ADJUST

    # URL of the Audio2Face Streaming Audio Player server (where A2F App is running)
    a2f_url = '172.16.15.209:50051'  # ADJUST to where the Audio2Face instance is running 

    # Prim path of the Audio2Face Streaming Audio Player on the stage (were to push the audio data)
    instance_name = '/World/audio2face/audio_player_streaming'     # streaming player for live sync 
    
    auth = riva.client.Auth(uri=riva_uri)
    tts_service = riva.client.SpeechSynthesisService(auth)

    language_code = 'en-US'
    sample_rate = 44100

    
    resp = tts_service.synthesize(question, language_code=language_code , sample_rate_hz=sample_rate , encoding=riva.client.AudioEncoding.LINEAR_PCM , voice_name="English-US.Female-1" )

    
    audio_data = np.frombuffer(resp.audio, dtype=np.int16)
    dtype = np.dtype('float32')
    i = np.iinfo(audio_data.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    audio_data = (audio_data.astype(dtype) - offset) / abs_max

    print(f"Sleeping for {sleep_time} seconds")
    time.sleep(sleep_time)
    push_audio_track(a2f_url, audio_data, sample_rate, instance_name)
    return f"Audio pushed to A2F"

def elevenlabs_api(text):
    print(f"text from rasa to elevn labs  {type(text)} {text}")
    logging.info(f"text from rasa to elevn labs  {type(text)}  {text}")
    str(text)
    
    voice_id= "XrExE9yKIg1WjnnlVkGX"
    CHUNK_SIZE = 1024
    model_id = "eleven_turbo_v2"
    url_eleven_labs = "https://api.elevenlabs.io/v1/text-to-speech/XrExE9yKIg1WjnnlVkGX"
    try:
        payload = {
        "model_id": model_id,
        # "pronunciation_dictionary_locators": [
        #     {
        #         "pronunciation_dictionary_id": "<string>",
        #         "version_id": "<string>"
        #     }
        # ],
        "text": text,
        "voice_settings": {
            "similarity_boost": 0.5,
            "stability": 0.75,
            # "style": 123,
            "use_speaker_boost": True
            }
        }
        # response = requests.request("POST", url=url_eleven_labs, json=payload, headers=headers)
        response = requests.post(url=url_eleven_labs, json=payload, headers=headers)
        logging.info(f"response {response}, {response.json()}")
        print(f"response {response}, {response.json()}")
        
    except Exception as e:
        logging.info(f"the error is: {e}") 
        
    with open('output.mp3', 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                
    mp3_file = AudioSegment.from_file("output.mp3", format="mp3")
    wav_file = mp3_file.set_frame_rate(44100).set_channels(1)
    wav_file.export("output.wav", format="wav")
    # os.remove('output.mp3')

def a2f_api_call():
    url_play_track = 'http://localhost:8011/A2F/Player/Play'
    url_set_current_track = 'http://localhost:8011/A2F/Player/SetTrack'
    
    
    body_set_track = {
        "a2f_player": a2f_player_regular,
        "file_name": 'output.wav',
        "time_range": [
             0,
            -1
        ]
    }
    
    response_set_track = requests.post(url=url_set_current_track , json=body_set_track)
    pprint.pprint(response_set_track.json())
    
    body = {
        "a2f_player": a2f_player_regular
    }
    response = requests.post(url=url_play_track , json=body)
    pprint.pprint(f" play track {response.json()}")
    
     
    
def sendtorasa(response):
    res = {
        "sender" : "test",
        "message" : response
    }       
    print("Sending Response to rasa")
    rasa_output = requests.post(url="http://172.16.15.209:5005/webhooks/rest/webhook", json=res).json()
    print(f'response from rasa {rasa_output}')
    return rasa_output[0]['text'], rasa_output[1]['custom']['action']

manager = ConnectionManager()
        
@app.get("/")
async def get():
    return f"server is live !"

# @app.post("/rag")
# def get_rag_response(item :Item):
#     response = get_query_response(item.text)
#     print(f"{response}")
#     return response

@app.post("/position{player_id}")
async def get_completion_response(data : StructData):
    data = data.dict()
    print(f"{data}")
    logging.info(f"{data}")
    return 


@app.post("/data")
async def get_completion_response(data: Item):
    # data = await data.json()
    # print(f"{data}")
    print(f"{data.text}")
    logging.info(f"{data}")
    # logging.info(f"{data.text}")
    return 


@app.post("/current_object")
async def get_completion_response(data : StructData):
    # data = data.dict()
    print(f"{data}")
    logging.info(f"{data}")
    return  


@app.websocket("/ws/bytes")
async def websocket_endpoint_bytes(websocket: WebSocket):
    await manager.connect(websocket)
    first_action = 'action=no-action?item_id=null?text=Hello, Welcome to Turtle AR. Let me know if you need help!'
    first_text = 'Hello, Welcome to Turtle AR are. Let me know if you need help!'
    await manager.send_personal_message(f"{first_action}", websocket)
    # elevenlabs_api(first_text)
    # a2f_api_call()
    print(convertToAudioAndPlay(first_text , 'en-US'))
    try:
        while True:
            data = await websocket.receive_bytes()
            print(f"message from UE client  :  type : {type(data)} , length  : {len(data)} ")
            # await manager.send_personal_message(f"You audio was recieved in {type(data)} format", websocket)
                
                
            auth = riva.client.Auth(uri=riva_uri)
            asr_service = riva.client.ASRService(auth)

            offline_config = riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                max_alternatives=1,
                enable_automatic_punctuation=False,
                verbatim_transcripts=False,
                profanity_filter=False,
                sample_rate_hertz=16000,
                audio_channel_count=1,
                language_code=language_code
            )
            riva.client.add_word_boosting_to_config(
            offline_config, boosted_words, boosted_lm_score)
            try:
                
                response = asr_service.offline_recognize(data, offline_config)
             
                if (len(response.results[0].alternatives) <= 0):
                    await manager.send_personal_message(f"No Audio recieved", websocket)
                else:

                    final_response = ""
                    for resp in response.results:
                        final_response = final_response + resp.alternatives[0].transcript
                    print(final_response)
                    
                    
                    # ===============================
                    rasa_text, rasa_action = sendtorasa(final_response)
                    print(f"The text sent from rasa {rasa_text}")
                    print(f"The action sent from rasa {rasa_action}")
                    await manager.send_personal_message(f"{rasa_action}", websocket)
                    # ===============================
                    # elevenlabs_api(rasa_text)
                    # # time.sleep(3)
                    # a2f_api_call()
                    print(convertToAudioAndPlay(rasa_text , 'en-US'))
            except Exception as e:
                print(f"the following exception occured : {e}")
                await manager.send_personal_message(f"No Audio recieved", websocket)            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.send_personal_message(f"Socket disconnected", websocket)
        