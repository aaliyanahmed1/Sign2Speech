import os
import requests
import pyttsx3

OLLAMA_URL = "http://localhost:11434/api/generate"
VOICE_DIR = "voices"

def generate_sentence_with_ollama(word):
    context_prompts = {
        'school': 'about education or learning',
        'sorry': 'about apologizing',
        'help': 'about assistance or support',
        'easy': 'about simplicity or convenience',
        'work': 'about occupation or tasks',
        'age': 'about time or years',
        'effort': 'about trying or working hard',
        'respect': 'about showing consideration',
        'near': 'about proximity or location',
        'home': 'about residence or family',
        'friend': 'about companionship',
        'washroom': 'about facilities',
        'preset': 'about settings or configurations',
        'pass': 'about success or approval',
        'fail': 'about unsuccessful attempts',
        'village': 'about community or rural life',
        'eating': 'about food or meals',
        'drinking': 'about beverages or hydration',
        'teacher': 'about education or guidance',
        'dress': 'about clothing or fashion',
        'message': 'about communication',
        'good': 'about positive qualities'
    }
    
    context = context_prompts.get(word.lower(), '')
    prompt = f"Generate a natural, everyday sentence using the word '{word}'{' ' + context if context else ''}. Keep it simple and conversational."
    
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        print(f"Error communicating with Ollama for '{word}': {e}")
        return f"{word.capitalize()}."

def text_to_speech(sentence, filename):
    engine = pyttsx3.init()
    engine.save_to_file(sentence, filename)
    engine.runAndWait()

def yolo_classes_to_voice(detected_classes):
    os.makedirs(VOICE_DIR, exist_ok=True)
    for cls in detected_classes:
        sentence = generate_sentence_with_ollama(cls)
        print(f"Class: {cls} | Sentence: {sentence}")
        audio_path = os.path.join(VOICE_DIR, f"{cls}.wav")
        text_to_speech(sentence, audio_path)
        print(f"Saved: {audio_path}")

# Example usage:
if __name__ == "__main__":
    detected = ["good", "help", "thank_you"]
    yolo_classes_to_voice(detected)