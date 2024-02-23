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
from tensor_check import get_query_response  , get_chat_response #,  get_rag_chat_response
from pydantic import BaseModel


app = FastAPI()
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s'
)
class Item(BaseModel):
    text: str

boosted_words = ["swift", "vision", "aqua", "gaze", "glasses", "mind", "lens"]
boosted_lm_score = 20.0
riva_uri = '172.16.15.209:50051'

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
        # audio_data.astype(np.float32).tobytes()
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

def push_audio_track_stream(url, audio_data, samplerate, instance_name):
    """
    This function pushes audio chunks sequentially via PushAudioStreamRequest()
    The function emulates the stream of chunks, generated by splitting input audio track.
    But in a real application such stream of chunks may be aquired from some other streaming source.
    The first message must contain start_marker field, containing only meta information (without audio data):
     * samplerate: sampling rate for the audio data
     * instance_name: prim path of the Audio2Face Streaming Audio Player on the stage, were to push the audio data
     * block_until_playback_is_finished: if True, the gRPC request will be blocked until the playback of the pushed track is finished (after the last message)
    Second and other messages must contain audio_data field:
     * audio_data: bytes, containing audio data for an audio chunk, where each sample is encoded as 4 bytes (float32)
    All messages are packed into a Python generator and passed to PushAudioStream()
    """

    chunk_size = samplerate // 10  # ADJUST
    sleep_between_chunks = 0.04  # ADJUST
    block_until_playback_is_finished = True  # ADJUST

    with grpc.insecure_channel(url) as channel:
        print("Channel creadted")
        stub = audio2face_pb2_grpc.Audio2FaceStub(channel)

        def make_generator():
            start_marker = audio2face_pb2.PushAudioRequestStart(
                samplerate=samplerate,
                instance_name=instance_name,
                block_until_playback_is_finished=block_until_playback_is_finished,
            )
            # At first, we send a message with start_marker
            yield audio2face_pb2.PushAudioStreamRequest(start_marker=start_marker)
            # Then we send messages with audio_data
            for i in range(len(audio_data) // chunk_size + 1):
                time.sleep(sleep_between_chunks)
                chunk = audio_data[i * chunk_size : i * chunk_size + chunk_size]
                yield audio2face_pb2.PushAudioStreamRequest(audio_data=chunk.astype(np.float32).tobytes())

        request_generator = make_generator()
        print("Sending audio data...")
        response = stub.PushAudioStream(request_generator)
        if response.success:
            print("SUCCESS")
        else:
            print(f"ERROR: {response.message}")
    print("Channel closed")

def bytes_to_audio(byte_data, audio_file):
    with wave.open(audio_file, 'wb') as wave_file:
        wave_file.setnchannels(1)  # Mono audio
        wave_file.setsampwidth(2)  # 2 bytes per sample
        wave_file.setframerate(16000)  # Standard audio sample rate

        wave_file.writeframes(byte_data)
        print(f"Audio file saved as {audio_file}")
        
audio_file = 'audio_from_unreal.wav'

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

    # if len(sys.argv) < 2:
    #     print("Format: python test_client.py PATH_TO_WAV INSTANCE_NAME")
    #     return

    # Sleep time emulates long latency of the request
    sleep_time = 0.2  # ADJUST

    # URL of the Audio2Face Streaming Audio Player server (where A2F App is running)
    a2f_url = '172.16.15.184:50051'  # ADJUST
    # rasa_url = "http://192.168.18.147:5005/webhooks/rest/webhook"
    # Local input WAV file path
    #audio_fpath = sys.argv[1]

    # Prim path of the Audio2Face Streaming Audio Player on the stage (were to push the audio data)
    instance_name = '/World/audio2face/audio_player_streaming'
    #'/World/audio2face/PlayerStreaming'
    # '/sk/LazyGraph/PlayerStreaming'
    
    # riva_uri = '172.16.15.209:50051'
    auth = riva.client.Auth(uri=riva_uri)
    tts_service = riva.client.SpeechSynthesisService(auth)

    language_code = 'en-US'
    sample_rate = 44100
    nchannels = 1
    sampwidth = 2
    
    # #print("Lets chat! , Please type 'quit' to  stop")
    # sentence = {
    #     "sender":"test",
    #     "message":question}
    
    
    # text = requests.post(url = rasa_url , json = sentence)
    # text = text.json()
    # print(text)
    # answer = text[0]["text"]
    # #print(answer)
    resp = tts_service.synthesize(question, language_code=language_code , sample_rate_hz=sample_rate , encoding=riva.client.AudioEncoding.LINEAR_PCM , voice_name="English-US.Female-1" )
    

    audio = resp.audio
    meta = resp.meta
    # print(meta)
    audio_file = "audio_from_unreal.wav"
    with wave.open(offline_output_file, 'wb') as out_f:
        out_f.setnchannels(nchannels)
        out_f.setsampwidth(sampwidth)
        out_f.setframerate(sample_rate)
        out_f.writeframesraw(resp.audio)
    
    audio_data = np.frombuffer(resp.audio, dtype=np.int16)
    dtype = np.dtype('float32')
    i = np.iinfo(audio_data.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    audio_data = (audio_data.astype(dtype) - offset) / abs_max

    # data, samplerate = soundfile.read(audio_file, dtype="float32")
    # # data = audio
    # # samplerate = sample_rate
    # print(f"sample rate : {samplerate}")
    # # Only Mono audio is supported
    # if len(data.shape) > 1:
    #     data = np.average(data, axis=1)

    print(f"Sleeping for {sleep_time} seconds")
    time.sleep(sleep_time)

    #if 0:  # ADJUST
        # Push the whole audio track at once
    push_audio_track(a2f_url, audio_data, sample_rate, instance_name)
    
    # push_audio_track(a2f_url, audio, samplerate, instance_name)
    
    
    return f"Audio pushed to A2F"
    #else:
        # Emulate audio stream and push audio chunks sequentially
        #push_audio_track_stream(url, data, samplerate, instance_name)

def sendtorasa(response):
    res = {
        "sender" : "test",
        "message" : response
    }       
    print("Sending Response to rasa")
    rasa_output = requests.post(url="http://172.16.15.216:5005/webhooks/rest/webhook", json=res).json()
    print(f'response from rasa {rasa_output}')
    return rasa_output[0]['text'], rasa_output[1]['custom']['action']

manager = ConnectionManager()
        
@app.get("/")
async def get():
    return HTMLResponse(html)

@app.post("/rag")
def get_rag_response(item :Item):
    print(f" type of request {type(item)} DATA : {item}")
    # data = await data.json()
    # print(data)
    print(item.text)
    response = get_query_response(item.text)
    print(f"{response}")
    
    return response

@app.post("/chat")
async def get_completion_response(data : Request):
    print(data)
    data = await data.json()
    print(data)
    response = get_chat_response(data['text'])
    print(f"{response}")
    
    return response

# @app.post("/chatrag")
# async def get_rag_chat_response(data : Request):
#     print(data)
#     print(data)
#     data = await data.json()
#     print(data)
#     response = get_rag_chat_response(data['text'])
#     print(f"{response}")
    
#     return response

@app.websocket("/ws/bytes")
async def websocket_endpoint_bytes(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_bytes()
            print(f"message from UE client  :  type : {type(data)} , length  : {len(data)} ")
            await manager.send_personal_message(f"You audio was recieved in {type(data)} format", websocket)
            bytes_to_audio(byte_data=data , audio_file=audio_file)
            with wave.open(audio_file, 'rb') as wav_file:
    # Read frames and convert to bytes data
                audio_data = wav_file.readframes(wav_file.getnframes())
                
                
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
                
                response = asr_service.offline_recognize(audio_data, offline_config)
                print(response)
                
                if (len(response.results[0].alternatives) <= 0):
                    await manager.send_personal_message(f"No Audio recieved", websocket)
                else:

                    final_response = ""
                    for resp in response.results:
                        final_response = final_response + resp.alternatives[0].transcript
                    # response = response.results[0].alternatives[0].transcript
                    print(final_response)
                    
                    
                    # ===============================
                    rasa_text, rasa_action = sendtorasa(final_response)
                    print(f"The text sent from rasa {rasa_text}")
                    print(f"The action sent from rasa {rasa_action}")
                    await manager.send_personal_message(f"{rasa_action}", websocket)
                    # ===============================
                    print(convertToAudioAndPlay(rasa_text , 'en-US'))
                    # await manager.send_personal_message(f"{rasa_action}", websocket)
            except Exception as e:
                print(f"the following exception occured : {e}")
                await manager.send_personal_message(f"No Audio recieved", websocket)
            
                
            # print(convertToAudioAndPlay(audio_data = data , question='response' , language = 'en-US'))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.send_personal_message(f"Socket disconnected", websocket)
        
        
@app.websocket("/ws/text")
async def websocket_endpoint_bytes(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:  
            data = await websocket.receive_text()
            print(f"message from UE client  :  type : {type(data)}") 
            # response = get_query_response(data)
            response = 'test tresponse'
            print(f"response from rasa {response}")
            await manager.send_personal_message(f"{response}", websocket)
            
              
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.send_personal_message(f"Socket disconnected", websocket)


@app.websocket("/ws/general")
async def websocket_endpoint_bytes(websocket: WebSocket):
    await manager.connect(websocket)  
    try:
        while True:  
            data = websocket.receive()
            print(f"message from UE client  :  type : {type(data)} , length  : {len(data)} ") 
            
              
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.send_personal_message(f"Socket disconnected", websocket)