import gradio as gr
import openai
import json

def get_openai_key(json_path):
    with open(json_path, "r") as f:
        keys = json.load(f)
    return keys["api_key"]

openai_key = get_openai_key("key.json")
openai.api_key = openai_key

transcription_text = None

def transcribe(audio):
    global transcription_text, message_history
    file = open(audio, "rb")
    transcription = openai.Audio.transcribe("whisper-1", file)

    transcription_text = transcription["text"]

    message_history =[
        {"role": "assistant", "content": f"Transkribering: {transcription_text}"},
        {"role": "system", "content": "Du är en journalassistent inom hälso-och sjukvården."}]

    response = openai.ChatCompletion.create(
        model= "gpt-3.5-turbo",
        messages=message_history
    )

    AImessage = response["choices"][0]["message"]["content"]
    message_history.append({"role": "assistant", "content": AImessage})
    return transcription_text

def chat_dialog(input):
    global message_history
    message_history.append({"role":"user", "content": input})

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = message_history
    )

    reply_content = completion.choices[0].message.content

    message_history.append({"role": "user", "content": reply_content})

    response = [(message_history[i]["content"], message_history[i +1]["content"]) for i in range (3, len(message_history) -1, 2)]

    print(message_history)
    return response

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    audio_file = gr.Audio(label = "Uppladdning av ljudfil",type='filepath')
    trans_output = gr.outputs.Textbox(label="Transkribering av dikterad journal")
    transcribe_btn = gr.Button("Transkribera uppladdad ljudfil")
    transcribe_btn.click(fn=transcribe, inputs=audio_file, outputs=trans_output)

    chatbot = gr.Chatbot(label = "Chat konversation")

    with gr.Row():
        txt = gr.Textbox(label = "Chatta", placeholder="Skriv ditt meddelande här").style(container=False)
        txt.submit(chat_dialog, txt, chatbot)
        txt.submit(lambda: "", None, txt)

demo.launch(share=True)
