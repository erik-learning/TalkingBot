# TalkingBot
This bot can understand natural language and answer your questions in many languages, including English, French, and Russian.

How it works:

Input: Your voice message is converted into text using a speech-to-text (STT) model.

Translation: The text is translated into English (if needed) using a translation model.

Information retrieval: The translated text is compared with the texts in our database. If there is a data point that is similar enough to the query, it is output.

GPT question answering: If there is no similar data point in the database, the query is sent to GPT for processing.

Output: The answer is spoken using a text-to-speech (TTS) model.

Speech-to-text model (STT/ASR): Whisper (Vosk worked well too)
Large Language Model (LLM): GPT-3.5-turbo
Retriever: MPNet
Translator: Google Translator
Text-to-speech model (TTS): Silero
