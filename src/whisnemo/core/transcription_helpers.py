import torch


def transcribe(
    audio_file: str,
    language: str,
    model_name: str,
    compute_dtype: str,
    suppress_numerals: bool,
    device: str,
):
    import whisper

    from whisnemo.core.helpers import find_numeral_symbol_tokens, wav2vec2_langs

    whisper_model = whisper.load_model(model_name, device=device)

    suppress_tokens = None
    if suppress_numerals:
        tokenizer = whisper.tokenizer.get_tokenizer(
            whisper_model.is_multilingual,
            language=language,
            task="transcribe",
        )
        suppress_tokens = find_numeral_symbol_tokens(tokenizer)

    if language is not None and language in wav2vec2_langs:
        word_timestamps = False
    else:
        word_timestamps = True

    result = whisper_model.transcribe(
        audio_file,
        language=language,
        beam_size=5,
        word_timestamps=word_timestamps,
        suppress_tokens=suppress_tokens if suppress_tokens is not None else "-1",
        fp16=(device == "cuda"),
        verbose=False,
    )

    whisper_results = []
    for segment in result["segments"]:
        whisper_results.append(
            {
                "id": segment.get("id"),
                "seek": segment.get("seek"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text"),
                "tokens": segment.get("tokens"),
                "temperature": segment.get("temperature"),
                "avg_logprob": segment.get("avg_logprob"),
                "compression_ratio": segment.get("compression_ratio"),
                "no_speech_prob": segment.get("no_speech_prob"),
                "words": segment.get("words"),
            }
        )

    detected_language = result.get("language", language)

    del whisper_model
    if device == "cuda":
        torch.cuda.empty_cache()

    return whisper_results, detected_language


def transcribe_batched(
    audio_file: str,
    language: str,
    batch_size: int,
    model_name: str,
    compute_dtype: str,
    suppress_numerals: bool,
    device: str,
):
    import whisperx

    whisper_model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_dtype,
        asr_options={"suppress_numerals": suppress_numerals},
    )
    audio = whisperx.load_audio(audio_file)
    result = whisper_model.transcribe(audio, language=language, batch_size=batch_size)
    del whisper_model
    if device == "cuda":
        torch.cuda.empty_cache()
    return result["segments"], result["language"], audio
