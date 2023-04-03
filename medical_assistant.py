import os
import json
import openai
#import UI_journal_assistant


def transcribe_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]

json_path = "key.json"
with open(json_path, "r") as f:
    keys = json.load(f)
openai_key = keys["api_key"]


def get_openai_key(json_path):
    with open(json_path, "r") as f:
        keys = json.load(f)
    return keys["api_key"]

openai.api_key = openai_key


def generate_journal_summary(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Du är en journalassistent inom hälso-och sjukvården."},
            {"role": "user", "content": f"Summanfatta innehåll av text: {text}"},
        ],
    )
    return response.choices[0].message.content


def chat_with_journal_assistant(text):
    conversations = [{"role": "user", "content": f"Summanfatta innehåll av text: {text}"}]

    while True:
        content = input("Vårdpersonal: ")
        conversations.append({"role": "user", "content": content})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversations,
        )

        chat_response = response.choices[0].message.content
        print(f'AI-Journalassistent: {chat_response}\nFinish reason for message: {response.choices[0].finish_reason}\n')
        conversations.append({"role": "assistant", "content": chat_response})


def main():
    audio_path = "cancer_meeting.m4a"
    text = transcribe_audio(audio_path)

    print("Transkriberar journal...")
    print(f'Journalanteckning: {text}\n')

    openai_key = get_openai_key("key.json")
    openai.api_key = openai_key

    print("Inväntar journalassistent...\n")

    journal_summary = generate_journal_summary(text)
    print(f'AI-Journalassistent: {journal_summary}\n')

    chat_with_journal_assistant(text)


if __name__ == "__main__":
    main()
