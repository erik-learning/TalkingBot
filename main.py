TOKEN = '**********************************'

from IPython.display import Audio, display
import telebot
import soundfile as sf
from faster_whisper import WhisperModel
import whisper
from get import retrieve_answers, ask_gpt
import torch
import os
from googletrans import Translator, LANGUAGES


bot = telebot.TeleBot(TOKEN)

model_stt = whisper.load_model("small")

translator = Translator()
def translate_to_russian(text: str) -> str:
    # Detect the language of the text
    detected_lang = translator.detect(text).lang

    # If the detected language is not Russian, translate it to Russian
    if detected_lang != 'ru':
        translated_text = translator.translate(text, src=detected_lang, dest='ru').text
        return translated_text
    return text


@bot.message_handler(content_types=['text'])
def handle_text(message):
    bot.reply_to(message, "Пришли нам аудио")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    photo_file_info = bot.get_file(message.photo[-1].file_id)  # Get the highest resolution photo
    downloaded_file = bot.download_file(photo_file_info.file_path)
    photo_path = 'received_photo.jpg'
    with open(photo_path, 'wb') as photo:
        photo.write(downloaded_file)

    with open(photo_path, 'rb') as photo:
        bot.send_photo(message.chat.id, photo)

@bot.message_handler(content_types=['voice'])
def handle_audio(message):
    audio_file_info = bot.get_file(message.voice.file_id)
    audio_path_ogg = 'received_audio.ogg'
    downloaded_file = bot.download_file(audio_file_info.file_path)

    with open(audio_path_ogg, 'wb') as audio:
        audio.write(downloaded_file)

    audio_path_wav = convert_audio_to_wav(audio_path_ogg)

    transcribed_text = process_audio_for_stt(audio_path_wav)
    print(transcribed_text)
    t1 = retrieve_answers(transcribed_text)

    if len(t1) == 0:
        t1 = ask_gpt(transcribed_text)
    else:
        t1 = t1[0]
        bot.reply_to(message, t1)

    # Translate the transcribed text to Russian if it's in another language
    t1 = translate_to_russian(t1)

    device = torch.device('cpu')
    torch.set_num_threads(4)
    local_file = 'model.pt'

    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v4_ru.pt',
                                       local_file)

    model_tts = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model_tts.to(device)
    sample_rate = 48000
    speaker = 'baya'
    audio_paths = model_tts.save_wav(text=t1,
                                 speaker=speaker,
                                 sample_rate=sample_rate)
    with open(audio_paths, 'rb') as audio_file:
        bot.send_audio(message.chat.id, audio_file)


def convert_audio_to_wav(audio_path_ogg: str) -> str:
    # Convert OGG to WAV
    audio_path_wav = 'converted_audio.wav'
    data, samplerate = sf.read(audio_path_ogg)
    sf.write(audio_path_wav, data, samplerate)
    return audio_path_wav

def process_audio_for_stt(audio_path_wav: str) -> str:
    result = model_stt.transcribe(audio_path_wav)
    return result["text"]

if __name__ == '__main__':
    bot.polling(none_stop=True)
