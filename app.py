import streamlit as st
import os
import wget
from omegaconf import OmegaConf
import json
import shutil
import torch
import torchaudio
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
import re
import logging
import nltk
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from matplotlib.patches import Rectangle
from ctc_forced_aligner import (
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)
import faster_whisper
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from matplotlib.patches import Rectangle
# ... (All your helper functions from the original script)

#represents the language for which a specific punctuation model is available.
punct_model_langs = [
    "en",
    "fr",
    "de",
    "es",
    "it",
    "nl",
    "pt",
    "bg",
    "pl",
    "cs",
    "sk",
    "sl",
]
LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
}
# maps 2 letter represent of language to 2 letter
langs_to_iso = {
    "af": "afr",
    "am": "amh",
    "ar": "ara",
    "as": "asm",
    "az": "aze",
    "ba": "bak",
    "be": "bel",
    "bg": "bul",
    "bn": "ben",
    "bo": "tib",
    "br": "bre",
    "bs": "bos",
    "ca": "cat",
    "cs": "cze",
    "cy": "wel",
    "da": "dan",
    "de": "ger",
    "el": "gre",
    "en": "eng",
    "es": "spa",
    "et": "est",
    "eu": "baq",
    "fa": "per",
    "fi": "fin",
    "fo": "fao",
    "fr": "fre",
    "gl": "glg",
    "gu": "guj",
    "ha": "hau",
    "haw": "haw",
    "he": "heb",
    "hi": "hin",
    "hr": "hrv",
    "ht": "hat",
    "hu": "hun",
    "hy": "arm",
    "id": "ind",
    "is": "ice",
    "it": "ita",
    "ja": "jpn",
    "jw": "jav",
    "ka": "geo",
    "kk": "kaz",
    "km": "khm",
    "kn": "kan",
    "ko": "kor",
    "la": "lat",
    "lb": "ltz",
    "ln": "lin",
    "lo": "lao",
    "lt": "lit",
    "lv": "lav",
    "mg": "mlg",
    "mi": "mao",
    "mk": "mac",
    "ml": "mal",
    "mn": "mon",
    "mr": "mar",
    "ms": "may",
    "mt": "mlt",
    "my": "bur",
    "ne": "nep",
    "nl": "dut",
    "nn": "nno",
    "no": "nor",
    "oc": "oci",
    "pa": "pan",
    "pl": "pol",
    "ps": "pus",
    "pt": "por",
    "ro": "rum",
    "ru": "rus",
    "sa": "san",
    "sd": "snd",
    "si": "sin",
    "sk": "slo",
    "sl": "slv",
    "sn": "sna",
    "so": "som",
    "sq": "alb",
    "sr": "srp",
    "su": "sun",
    "sv": "swe",
    "sw": "swa",
    "ta": "tam",
    "te": "tel",
    "tg": "tgk",
    "th": "tha",
    "tk": "tuk",
    "tl": "tgl",
    "tr": "tur",
    "tt": "tat",
    "uk": "ukr",
    "ur": "urd",
    "uz": "uzb",
    "vi": "vie",
    "yi": "yid",
    "yo": "yor",
    "yue": "yue",
    "zh": "chi",
}
#creates a list called whisper_langs by sorting and combining language codes from LANGUAGES and TO_LANGUAGE_CODE
whisper_langs = sorted(LANGUAGES.keys()) + sorted(
    [k.title() for k in TO_LANGUAGE_CODE.keys()]  #iterates through language name from TO_LANGUAGE_CODES.keys(). .title() capitalises first letter of each word in the string
)

#Generates configuration file for speaker diaraization
def create_config(output_dir):
    DOMAIN_TYPE = "telephonic"  # Can be meeting, telephonic, or general based on domain type of the audio file
    CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
    CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"
    MODEL_CONFIG = os.path.join(output_dir, CONFIG_FILE_NAME) #Creates path to downloaded configuration file
    if not os.path.exists(MODEL_CONFIG):
        MODEL_CONFIG = wget.download(CONFIG_URL, output_dir)

    config = OmegaConf.load(MODEL_CONFIG) #Loads the downloaded congiguration file. Likely to specify framework i,.e NEMO

    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    meta = {
        "audio_filepath": os.path.join(output_dir, "mono_file.wav"), #will be converted into single audio channel, stores in lossless format; mono audio - reduces complexity, focuses on speech amd compatibility
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    with open(os.path.join(data_dir, "input_manifest.json"), "w") as fp:
        json.dump(meta, fp)
        fp.write("\n")

    pretrained_vad = "vad_multilingual_marblenet"
    pretrained_speaker_model = "titanet_large"
    config.num_workers = 0  # Workaround for multiprocessing hanging with ipython issue
    config.diarizer.manifest_filepath = os.path.join(data_dir, "input_manifest.json")
    config.diarizer.out_dir = (
        output_dir  # Directory to store intermediate files and prediction outputs
    )

    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.oracle_vad = (
        False  # compute VOICE ACTIVITY DETECTION provided with model_path to vad config
    )
    config.diarizer.clustering.parameters.oracle_num_speakers = False

    # Here, we use our in-house pretrained NeMo VAD model
    config.diarizer.vad.model_path = pretrained_vad
    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05
    
    config.diarizer.msdd_model.model_path = (
        "diar_msdd_telephonic"  # Telephonic speaker diarization model
    )
    
  

    return config




def visualize_audio_processing(original_audio_path, processed_audio_path, speaker_segments=None, output_dir=None):

    # Load audio files
    y_orig, sr_orig = librosa.load(original_audio_path, sr=None)
    y_proc, sr_proc = librosa.load(processed_audio_path, sr=None)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # Plot original waveform
    librosa.display.waveshow(y_orig, sr=sr_orig, ax=ax1, alpha=0.6)
    ax1.set_title('Original Audio Waveform')
    ax1.set_xlabel('')
    ax1.grid(True)
    
    # Plot processed waveform with speaker colors if available
    if speaker_segments:
        # Create base waveform plot
        librosa.display.waveshow(y_proc, sr=sr_proc, ax=ax2, alpha=0.3, color='gray')
        
        # Get unique speakers and assign colors
        speakers = sorted(list(set(spk for _, _, spk in speaker_segments)))
        colors = plt.cm.Set3(np.linspace(0, 1, len(speakers)))
        color_dict = dict(zip(speakers, colors))
        
        # Add colored segments for each speaker
        for start, end, speaker in speaker_segments:
            start_idx = int(start * sr_proc / 1000)  # Convert ms to samples
            end_idx = int(end * sr_proc / 1000)
            
            if start_idx < len(y_proc) and end_idx <= len(y_proc):
                segment = y_proc[start_idx:end_idx]
                t = np.linspace(start/1000, end/1000, len(segment))
                ax2.plot(t, segment, color=color_dict[speaker], alpha=0.7)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], color=color_dict[spk], 
                         label=f'Speaker {spk}') for spk in speakers]
        ax2.legend(handles=legend_elements, loc='upper right')
    else:
        # If no speaker segments, just show the processed waveform
        librosa.display.waveshow(y_proc, sr=sr_proc, ax=ax2, alpha=0.6)
    
    ax2.set_title('Processed Audio Waveform with Speaker Diarization')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save if output directory is provided
    if output_dir:
        plt.savefig(f"{output_dir}/waveform_visualization.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()

#Function used to determine a specific point in time within time segment that start with 's' and end time is 'e'.
def get_word_ts_anchor(s,e,option='start'): #option is a parameter with  default value start
    if option=='end':
        return e
    elif option =='mid':
        return (s+e)/2
    return s

#takes word timestamp and speaker timestamp as input and creates a mapping between each word and speaker
def get_words_speaker_mapping(wrd_ts,spk_ts,word_anchor_option='start'): # wrd_ts- lst of dict which a word contains start_time, end time and text; spk_ts= list of tuples, where each tuple represnts a speaker and has s,e,and speaker id(sp)
    s, e, sp = spk_ts[0]
    wrd_pos, turn_idx = 0, 0
    wrd_spk_mapping = []
    
    for i, wrd_dict in enumerate(wrd_ts):
        ws = int(wrd_dict['start'] * 1000)  # Convert to milliseconds
        we = int(wrd_dict['end'] * 1000)
        wrd = wrd_dict['text']
        
        wrd_pos = get_word_ts_anchor(ws, we, word_anchor_option)
        
        # Find the correct speaker turn
        while wrd_pos > float(e) and turn_idx < len(spk_ts) - 1:
            turn_idx += 1
            s, e, sp = spk_ts[turn_idx]
        
        # Ensure end time matches start time of next word if available
        next_start = None
        if i < len(wrd_ts) - 1:
            next_start = int(wrd_ts[i + 1]['start'] * 1000)
            we = next_start  # Set end time to start of next word
        
        wrd_spk_mapping.append({
            "word": wrd,
            "start_time": ws,
            "end_time": we,
            "speaker": sp
        })
    
    return wrd_spk_mapping
    
sentence_ending_punctuations= ".?!"

#aims to find the index of the first word of the sentence containing the word at a given index (word_idx). It considers speaker changes and a maximum word lookback limit. 
def get_first_word_idx_of_sentence(word_idx, word_list, speaker_list,max_words):
    is_word_sentence_end=(
        lambda x: x >=0 and word_list[x][-1] in sentence_ending_punctuations
    )
    left_idx = word_idx
    while (
        left_idx > 0
        and word_idx - left_idx < max_words#This prevents the function from searching through the entire list in very long texts without sentence endings.
        and speaker_list[left_idx - 1] == speaker_list[left_idx] #checks if the speaker of the previous word is the same as the speaker of the current word
        and not is_word_sentence_end(left_idx - 1)
    ):
        left_idx -= 1

    return left_idx if left_idx == 0 or is_word_sentence_end(left_idx - 1) else -1

def get_last_word_idx_of_sentence(word_idx, word_list, max_words):
    is_word_sentence_end = (
        lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations
    )
    right_idx = word_idx
    while (
        right_idx < len(word_list) - 1
        and right_idx - word_idx < max_words
        and not is_word_sentence_end(right_idx)
    ):
        right_idx += 1

    return (
        right_idx
        if right_idx == len(word_list) - 1 or is_word_sentence_end(right_idx)
        else -1
    )
    
#This function realigns speaker labels in word-speaking mapping 
def get_realigned_ws_mapping_with_punctuation(
    word_speaker_mapping, max_words_in_sentence=50 # word_speaker_mapping is list of dict where it contains word- the spoken word and speaker- assigned speaker label
):
    is_word_sentence_end = (
        lambda x: x >= 0
        and word_speaker_mapping[x]["word"][-1] in sentence_ending_punctuations #used to detect boundaries
    )
    wsp_len = len(word_speaker_mapping)

    words_list, speaker_list = [], []
    for k, line_dict in enumerate(word_speaker_mapping):
        word, speaker = line_dict["word"], line_dict["speaker"]
        words_list.append(word)
        speaker_list.append(speaker)

    k = 0
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k]
        #checks if there's speaker change in the middle of the sentence
        if (
            k < wsp_len - 1
            and speaker_list[k] != speaker_list[k + 1]
            and not is_word_sentence_end(k) # current word is not the end of the sentence
        ):
            left_idx = get_first_word_idx_of_sentence(
                k, words_list, speaker_list, max_words_in_sentence
            )
            right_idx = (
                get_last_word_idx_of_sentence(
                    k, words_list, max_words_in_sentence - k + left_idx - 1
                )
                if left_idx > -1
                else -1
            )
            if min(left_idx, right_idx) == -1:
                k += 1
                continue

            spk_labels = speaker_list[left_idx : right_idx + 1] #slice of speaker_list from start to end
            mod_speaker = max(set(spk_labels), key=spk_labels.count) #determines the most frequent speaker
            if spk_labels.count(mod_speaker) < len(spk_labels) // 2: # count of the majority speaker is less than half the total number of labels in this segment- speaker distribution is too inconsistent
                k += 1 # moves to next word
                continue

            speaker_list[left_idx : right_idx + 1] = [mod_speaker] * (
                right_idx - left_idx + 1
            )
            k = right_idx

        k += 1

    k, realigned_list = 0, []
    while k < len(word_speaker_mapping):
        line_dict = word_speaker_mapping[k].copy() # it has word and its original speaker label, .copy() ensures the dictionary is not modified 
        line_dict["speaker"] = speaker_list[k] #Updated speaker list, contains the realigned speaker label 
        realigned_list.append(line_dict) 
        k += 1

    return realigned_list

# groups words into sentences , assigning each sentence to a speaker
def get_sentences_speaker_mapping(word_speaker_mapping, spk_ts):
    
    
    if not word_speaker_mapping:
        return []
    
    sentences = []
    current_sentence = {
        "speaker": None,
        "start_time": word_speaker_mapping[0]["start_time"],
        "end_time": None,
        "text": ""
    }
    
    last_speaker = None
    
    for i, word_dict in enumerate(word_speaker_mapping):
        current_speaker = f"Speaker {word_dict['speaker']}"
        word = word_dict["word"]
        
        # Start a new sentence only if speaker changes
        if last_speaker is not None and current_speaker != last_speaker:
            # Set the end time to the current word's start time
            current_sentence["end_time"] = word_dict["start_time"]
            
            if current_sentence["text"].strip():
                sentences.append(dict(current_sentence))
            
            # Start new sentence
            current_sentence = {
                "speaker": current_speaker,
                "start_time": word_dict["start_time"],
                "end_time": None,
                "text": ""
            }
        
        # Initialize speaker for first sentence
        if current_sentence["speaker"] is None:
            current_sentence["speaker"] = current_speaker
        
        current_sentence["text"] += word + " "
        last_speaker = current_speaker
        
        # Handle the last word
        if i == len(word_speaker_mapping) - 1:
            current_sentence["end_time"] = word_dict["end_time"]
            if current_sentence["text"].strip():
                sentences.append(dict(current_sentence))
    
    # Clean up the text by removing extra spaces
    for sentence in sentences:
        sentence["text"] = sentence["text"].strip()
    
    return sentences
    
                


#Formats timestamp in HH:MM:SS
def format_timestamp(milliseconds: float, always_include_hours: bool = False, decimal_marker: str = "."):
     assert milliseconds >= 0, "non-negative timestamp expected"

     hours = milliseconds // 3_600_000
     milliseconds -= hours * 3_600_000

     minutes = milliseconds // 60_000
     milliseconds -= minutes * 60_000

     seconds = milliseconds // 1_000
     milliseconds -= seconds * 1_000

     hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else "00:"
     return (
         f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
     )
    

def get_speaker_aware_transcript(sentences_speaker_mapping, f, include_timestamps=False):
    """
    Takes a list of sentences with speaker information and writes a formatted
    transcript to a file, separating paragraphs by the speaker and including
    timestamps if requested.
    """

    if not sentences_speaker_mapping:
        return
    
    def format_time(ms):
        """Format milliseconds to HH:MM:SS.mmm"""
        seconds = ms / 1000
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    
    last_end_time = 0
    last_speaker = None
    
    for i, sentence in enumerate(sentences_speaker_mapping):
        # Ensure continuous timing
        start_time = max(sentence["start_time"], last_end_time)
        end_time = sentence["end_time"]
        
        if i < len(sentences_speaker_mapping) - 1:
            # Make sure this sentence's end time matches next sentence's start time
            next_start = sentences_speaker_mapping[i + 1]["start_time"]
            end_time = next_start
            
        # Add extra line break if speaker changes
        current_speaker = sentence['speaker']
        if last_speaker is not None and current_speaker != last_speaker:
            f.write("\n")
        
        # Format the line
        if include_timestamps:
            timestamp = f"[{format_time(start_time)} - {format_time(end_time)}]"
            f.write(f"{sentence['speaker']} {timestamp}: {sentence['text'].strip()}\n")
        else:
            f.write(f"{sentence['speaker']}: {sentence['text'].strip()}\n")
        
        last_end_time = end_time
        last_speaker = current_speaker

#Takes transcript and writes it into a SubRip Subtitle format(SRT), common for subtitles and captioning
def write_str(transcript,file):
    for i, segment in enumerate(transcript,start=1):
        #writes start lines
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start_time'], always_include_hours=True, decimal_marker=',')} --> "
            f"{format_timestamp(segment['end_time'], always_include_hours=True, decimal_marker=',')}\n"
            f"{segment['speaker']}: {segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )
       
def find_numeral_symbol_tokens(tokenizer):
    numeral_symbol_tokens= [-1,] # default value/ placeholder that will be included in the returned list regardless of whether any numeral or symbol tokens are found 
    for token, token_id in tokenizer.get_vocab().items():
        has_numeral_symbol= any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(token_id)
    return numeral_symbol_tokens


def _get_next_start_timestamp(word_timestamps, current_word_index, final_timestamp):
    # if current word is the last word
    if current_word_index == len(word_timestamps) - 1:
        return word_timestamps[current_word_index]["start"]

    next_word_index = current_word_index + 1
    while current_word_index < len(word_timestamps) - 1:
        if word_timestamps[next_word_index].get("start") is None:
            # if next word doesn't have a start timestamp
            # merge it with the current word and delete it
            word_timestamps[current_word_index]["word"] += (
                " " + word_timestamps[next_word_index]["word"]
            )

            word_timestamps[next_word_index]["word"] = None
            next_word_index += 1
            if next_word_index == len(word_timestamps):
                return final_timestamp

        else:
            return word_timestamps[next_word_index]["start"]
        
def filter_missing_timestamps(
    word_timestamps, initial_timestamp=0, final_timestamp=None
):
    # handle the first and last word
    if word_timestamps[0].get("start") is None:
        word_timestamps[0]["start"] = (
            initial_timestamp if initial_timestamp is not None else 0
        )
        word_timestamps[0]["end"] = _get_next_start_timestamp(
            word_timestamps, 0, final_timestamp
        )

    result = [word_timestamps[0],]
    for i, ws in enumerate(word_timestamps[1:], start=1):
        # if ws doesn't have a start and end
        # use the previous end as start and next start as end
        if ws.get("start") is None and ws.get("word") is not None:
            ws["start"] = word_timestamps[i - 1]["end"]
            ws["end"] = _get_next_start_timestamp(word_timestamps, i, final_timestamp)

        if ws["word"] is not None:
            result.append(ws)
    return result

def cleanup(path: str):
    """path could either be relative or absolute."""
    # check if file or directory exists
    if os.path.isfile(path) or os.path.islink(path):
        # remove file
        os.remove(path)
    elif os.path.isdir(path):
        # remove directory and all its content
        shutil.rmtree(path)
    else:
        raise ValueError("Path {} is not a file or dir.".format(path))
    
def process_language_arg(language: str, model_name: str):
    """
    Process the language argument to make sure it's valid and convert language names to language codes.
    """
    if language is not None:
        language = language.lower()
    if language not in LANGUAGES:
        if language in TO_LANGUAGE_CODE:
            language = TO_LANGUAGE_CODE[language]
        else:
            raise ValueError(f"Unsupported language: {language}")

    if model_name.endswith(".en") and language != "en":
        if language is not None:
            logging.warning(
                f"{model_name} is an English-only model but received '{language}'; using English instead."
            )
        language = "en"
    return language



def process_audio(audio_path, enable_stemming=True, whisper_model_name="small",
                  suppress_numerals=True, batch_size=8, language=None,
                  device="cuda" if torch.cuda.is_available() else "cpu"):
    """Processes the audio file using Demucs, Whisper, forced alignment, and NeMo diarization."""

    temp_path = os.path.join(os.getcwd(), "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)
    try:
        y_orig, sr_orig = librosa.load(audio_path, sr=None)

        if enable_stemming:
            return_code = os.system(
                f'python -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o "temp_outputs" --device "{device}"'
            )

            if return_code != 0:
                logging.warning("Source splitting failed, using original audio file.")
                vocal_target = audio_path
            else:
                vocal_target = os.path.join(
                    "temp_outputs",
                    "htdemucs",
                    os.path.splitext(os.path.basename(audio_path))[0],
                    "vocals.wav",
                )
        else:
            vocal_target = audio_path
        y_proc, sr_proc = librosa.load(vocal_target, sr=None)


        compute_type = "int8"
        whisper_model = faster_whisper.WhisperModel(
            whisper_model_name, device=device, compute_type=compute_type
        )
        whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
        audio_waveform = faster_whisper.decode_audio(vocal_target)
        suppress_tokens = (
            find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
            if suppress_numerals
            else [-1]
        )

        if batch_size > 0:
            transcript_segments, info = whisper_pipeline.transcribe(
                audio_waveform,
                language,
                suppress_tokens=suppress_tokens,
                batch_size=batch_size,
                without_timestamps=True,
            )
        else:
            transcript_segments, info = whisper_model.transcribe(
                audio_waveform,
                language,
                suppress_tokens=suppress_tokens,
                without_timestamps=True,
                vad_filter=True,
            )

        full_transcript = "".join(segment.text for segment in transcript_segments)

        del whisper_model, whisper_pipeline
        torch.cuda.empty_cache()

        alignment_model, alignment_tokenizer = load_alignment_model(
            device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        audio_waveform = (
            torch.from_numpy(audio_waveform)
            .to(alignment_model.dtype)
            .to(alignment_model.device)
        )

        emissions, stride = generate_emissions(alignment_model, audio_waveform, batch_size=batch_size)

        del alignment_model
        torch.cuda.empty_cache()

        tokens_starred, text_starred = preprocess_text(
            full_transcript,
            romanize=True,
            language=langs_to_iso[info.language],
        )

        segments, scores, blank_token = get_alignments(
            emissions,
            tokens_starred,
            alignment_tokenizer,
        )

        spans = get_spans(tokens_starred, segments, blank_token)

        word_timestamps = postprocess_results(text_starred, spans, stride, scores)

        torchaudio.save(
            os.path.join(temp_path, "mono_file.wav"),
            audio_waveform.cpu().unsqueeze(0).float(),
            16000,
            channels_first=True
        )
        config = create_config(temp_path)
        config.diarizer.msdd_model.parameters.kernel_size = (9, 1)
        msdd_model = NeuralDiarizer(cfg=config)
        msdd_model.diarize()

        speaker_ts = []
        with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(" ")
                s = int(float(line_list[5]) * 1000)
                e = s + int(float(line_list[8]) * 1000)
                speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

        if info.language in punct_model_langs:
            punct_model = PunctuationModel(model="kredor/punctuate-all")
            words_list = list(map(lambda x: x["word"], wsm))
            labled_words = punct_model.predict(words_list, chunk_size=230)
            ending_puncts = ".?!"
            model_puncts = ".,;:!?"
            is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

            for word_dict, labeled_tuple in zip(wsm, labled_words):
                word = word_dict["word"]
                if (
                    word
                    and labeled_tuple[1] in ending_puncts
                    and (word[-1] not in model_puncts or is_acronym(word))
                ):
                    word += labeled_tuple[1]
                    if word.endswith(".."):
                        word = word.rstrip(".")
                    word_dict["word"] = word

        else:
            logging.warning(
                f"Punctuation restoration is not available for {info.language} language. Using the original punctuation."
            )

        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)
        
        transcript_path = f"{os.path.splitext(audio_path)[0]}.txt"
        with open(transcript_path, "w", encoding="utf-8-sig") as f:
            get_speaker_aware_transcript(ssm, f, include_timestamps=True)
        
        # return temp_path, speaker_ts, transcript_path, audio_waveform, vocal_target, y_orig, sr_orig, y_proc, sr_proc
        return {
    'temp_path': temp_path,           # Path to temporary directory
    'speaker_ts': speaker_ts,         # Speaker timestamps
    'transcript_path': transcript_path, # Path to transcript file
    'audio_waveform': audio_waveform, # Processed audio waveform
    'vocal_target': vocal_target,     # Path to processed vocals
    'y_orig': y_orig,                 # Original audio data
    'sr_orig': sr_orig,               # Original sample rate
    'y_proc': y_proc,                 # Processed audio data
    'sr_proc': sr_proc                # Processed sample rate
}
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        return None

st.set_page_config(page_title="voice-sphere", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for light theme
st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    
    /* Custom styling for file uploader */
    .uploadedFile {
        border: 2px solid #4B9EF9;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        background-color: #F0F8FF;
    }
    
    /* Custom button styling */
    .stButton > button {
        background-color: #4B9EF9;
        color: white;
        padding: 0.5rem 1rem;
        font-size: 16px;
        font-weight: 500;
        border: none;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #357ABD;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* File uploader styling */
    .st-bw {
        background-color: #F8F9FA;
        border: 2px dashed #4B9EF9;
        border-radius: 5px;
        padding: 10px;
    }
    
    /* Container styling */
    .css-1d391kg {
        padding: 1rem;
        background-color: #FFFFFF;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Instructions box styling */
    .instructions-box {
        background-color: #F0F8FF;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4B9EF9;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)



st.title("🎙️ voice-sphere")

st.markdown("""
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
        <h4>📌 Upload your audio file to:</h4>
        <ul>
            <li>🎯 Separate speakers</li>
            <li>📝 Generate transcript</li>
            <li>📊 Visualize audio waves</li>
        </ul>
    </div>
""", unsafe_allow_html=True)



uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])

       
if uploaded_file is not None:
    st.markdown(f"""
        <div class="uploadedFile">
            📁 Uploaded: {uploaded_file.name}
        </div>
    """, unsafe_allow_html=True)

    # Display and play the uploaded audio
    st.subheader("🎵 Uploaded Audio")
    st.audio(uploaded_file, format='audio/wav')

    audio_path = "uploaded_audio." + uploaded_file.name.split(".")[-1]
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("🔄 Process Audio", key="process_button"):
        with st.spinner("🎵Processing audio..."):
            try:
                result = process_audio(audio_path)
                
                if result is None:
                    st.error("Audio processing failed. Please try again.")
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                    # return None

                # Visualization
                
                st.subheader("📊 Waveform Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("🎵 Original Audio")
                    fig_orig, ax_orig = plt.subplots(figsize=(6, 3))
                    librosa.display.waveshow(result['y_orig'], sr=result['sr_orig'], ax=ax_orig)
                    st.pyplot(fig_orig)

                with col2:
                    st.write("🎧 Processed Audio with Speaker Diarization")
                    fig_proc, ax_proc = plt.subplots(figsize=(6, 3))
                    librosa.display.waveshow(result['y_proc'], sr=result['sr_proc'], ax=ax_proc)
                    
                    if result['speaker_ts']:
                        speakers = sorted(list(set(spk for _, _, spk in result['speaker_ts'])))
                        colors = plt.cm.Set3(np.linspace(0, 1, len(speakers)))
                        color_dict = dict(zip(speakers, colors))
                        for start, end, speaker in result['speaker_ts']:
                            start_idx = int(start * result['sr_proc'] / 1000)
                            end_idx = int(end * result['sr_proc'] / 1000)
                            if start_idx < len(result['y_proc']) and end_idx <= len(result['y_proc']):
                                segment = result['y_proc'][start_idx:end_idx]
                                t = np.linspace(start/1000, end/1000, len(segment))
                                ax_proc.plot(t, segment, color=color_dict[speaker], alpha=0.7)
                        legend_elements = [plt.Line2D([0], [0], color=color_dict[spk], 
                                         label=f'Speaker {spk}') for spk in speakers]
                        ax_proc.legend(handles=legend_elements, loc='upper right')
                    st.pyplot(fig_proc)

                # Output Transcript
                st.subheader("📝 Diarized Transcript")
                if os.path.exists(result['transcript_path']):
                    with open(result['transcript_path'], "r", encoding="utf-8-sig") as f:
                        transcript_text = f.read()
                        st.markdown(f'''
                            <div style='background-color: #f0f2f6; padding: 20px; 
                                border-radius: 10px; font-family: monospace;'>
                                {transcript_text.replace(chr(10), '<br>')}
                            </div>
                        ''', unsafe_allow_html=True)

                # Cleanup
                cleanup(result['temp_path'])
                os.remove(audio_path)
                os.remove(result['transcript_path'])

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                if os.path.exists(audio_path):
                    os.remove(audio_path)





