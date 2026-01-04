"""
SAM Audio CLI - Command-line interface for text-prompted audio separation
Memory-optimized version using sam-audio-small by default

Usage:
    python sam_audio_cli.py --input audio.mp3 --prompt "vocals" --mode extract
    python sam_audio_cli.py  # Interactive mode
"""
import torch
import torchaudio
import gc
import os
import time
import argparse
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

load_dotenv()


def show_gpu_memory(label: str = ""):
    """Show current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory {label}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def clear_memory():
    """Clear GPU and system memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def create_lite_model(model_name: str = "facebook/sam-audio-small", token: Optional[str] = None):
    """
    Create a memory-optimized SAM Audio model by removing unused components.

    This reduces VRAM usage from ~11GB to ~4-6GB by:
    - Replacing vision_encoder with a dummy (saves ~2GB)
    - Disabling visual_ranker (saves ~2GB)
    - Disabling text_ranker (saves ~2GB)
    - Disabling span_predictor (saves ~1-2GB)

    Args:
        model_name: HuggingFace model name (default: facebook/sam-audio-small)
        token: HuggingFace access token (optional)

    Returns:
        model: Optimized SAM Audio model
        processor: SAM Audio processor
    """
    from sam_audio import SAMAudio, SAMAudioProcessor
    from huggingface_hub import login

    print(f"\n{'='*60}")
    print(f"Loading {model_name}...")
    print(f"{'='*60}")

    # Resolve token from (1) function arg, (2) env vars
    token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "No Hugging Face token provided. Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN or pass --token."
        )
    # Keep login (no interactive prompt, no git credential storage)
    login(token=token, add_to_git_credential=False)

    # Load model (token is now in HF cache)
    model = SAMAudio.from_pretrained(model_name)
    processor = SAMAudioProcessor.from_pretrained(model_name)

    print("\nOptimizing model for low VRAM...")

    # Get vision encoder dim before deleting
    vision_dim = model.vision_encoder.dim if hasattr(model.vision_encoder, 'dim') else 1024

    # Delete heavy components
    del model.vision_encoder
    gc.collect()
    print("  ✓ Removed vision_encoder")

    # Store the dim for _get_video_features
    model._vision_encoder_dim = vision_dim

    # Monkey patch _get_video_features to return zeros
    def dummy_get_video_features(self, video, video_mask=None):
        """Return dummy video features (zeros)"""
        batch_size = video.shape[0] if hasattr(video, 'shape') else 1
        device = next(self.parameters()).device
        return torch.zeros((batch_size, 1, self._vision_encoder_dim), device=device)

    model._get_video_features = dummy_get_video_features.__get__(model, type(model))
    print("  ✓ Patched video feature extraction")

    # Delete visual ranker
    if hasattr(model, 'visual_ranker') and model.visual_ranker is not None:
        del model.visual_ranker
        model.visual_ranker = None
        gc.collect()
        print("  ✓ Removed visual_ranker")

    # Delete text ranker (if not needed)
    if hasattr(model, 'text_ranker') and model.text_ranker is not None:
        del model.text_ranker
        model.text_ranker = None
        gc.collect()
        print("  ✓ Removed text_ranker")

    # Delete span predictor
    if hasattr(model, 'span_predictor') and model.span_predictor is not None:
        del model.span_predictor
        model.span_predictor = None
        gc.collect()
        print("  ✓ Removed span_predictor")

    # Delete span predictor transform
    if hasattr(model, 'span_predictor_transform') and model.span_predictor_transform is not None:
        del model.span_predictor_transform
        model.span_predictor_transform = None
        gc.collect()
        print("  ✓ Removed span_predictor_transform")

    # Force garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n✓ Model optimization complete!")

    return model, processor


def load_audio_chunks(audio_path: str, chunk_duration: float = 25.0, sample_rate: int = 44100):
    """Load audio file and split into chunks"""
    print(f"\nLoading audio: {audio_path}")
    waveform, sr = torchaudio.load(audio_path)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        sr = sample_rate

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Split into chunks
    chunk_samples = int(chunk_duration * sample_rate)
    total_samples = waveform.shape[1]
    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_samples
        end = min((i + 1) * chunk_samples, total_samples)
        chunk = waveform[:, start:end]

        # Pad last chunk if needed
        if chunk.shape[1] < chunk_samples:
            padding = chunk_samples - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, padding))

        chunks.append(chunk)

    print(f"Split into {num_chunks} chunks of {chunk_duration}s each")
    return chunks, sr, total_samples


def separate_audio(
    input_path: str,
    prompt: str,
    output_dir: str = "output",
    mode: str = "extract",
    model_size: str = "small",
    use_float32: bool = False,
    chunk_duration: float = 25.0,
    token: Optional[str] = None,
):
    """
    Separate audio using text prompt

    Args:
        input_path: Path to input audio file
        prompt: Text prompt describing target sound
        output_dir: Output directory
        mode: "extract" or "remove"
        model_size: Model size ("small" recommended for low VRAM)
        use_float32: Use float32 instead of bfloat16 (slower, more memory)
        chunk_duration: Chunk duration in seconds
        token: HuggingFace access token (optional)
    """
    clear_memory()
    show_gpu_memory("before model load")

    # Select model
    model_name = f"facebook/sam-audio-{model_size}"

    # Load optimized model
    model, processor = create_lite_model(model_name, token=token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if use_float32 else torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = model.to(device, dtype=dtype)
    model.eval()

    show_gpu_memory("after model load")

    # Load and chunk audio
    chunks, sample_rate, total_samples = load_audio_chunks(input_path, chunk_duration)

    # Process each chunk
    outputs = []
    with torch.inference_mode():
        for i, chunk in enumerate(chunks):
            print(f"\nProcessing chunk {i+1}/{len(chunks)}...")
            start_time = time.time()

            # Prepare inputs
            audio_input = chunk.squeeze().numpy()
            inputs = processor(
                audios=[audio_input],
                descriptions=[prompt],
                return_tensors="pt",
                sampling_rate=sample_rate
            )

            # Move to device
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            # Run inference
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=torch.cuda.is_available() and not use_float32):
                result = model.separate(inputs)

            # Get output audio
            separated = result.audio[0].cpu()
            outputs.append(separated)

            # Clear intermediate tensors
            del inputs, result
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            elapsed = time.time() - start_time
            print(f"Chunk processed in {elapsed:.1f}s")
            show_gpu_memory(f"after chunk {i+1}")

    # Concatenate outputs and trim to original length
    output_waveform = torch.cat(outputs, dim=-1)[:, :total_samples]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save output
    input_name = Path(input_path).stem
    output_path = os.path.join(output_dir, f"{input_name}_{mode}_{prompt.replace(' ', '_')}.wav")

    torchaudio.save(output_path, output_waveform, sample_rate)
    print(f"\n✓ Saved output to: {output_path}")

    # Cleanup
    del model, processor, outputs, output_waveform
    clear_memory()

    return output_path


def interactive_mode(token: Optional[str] = None):
    """Run in interactive mode with text prompts"""
    print(f"\n{'='*60}")
    print("SAM Audio CLI - Interactive Mode")
    print(f"{'='*60}\n")

    # Get input file
    input_path = input("Enter path to audio file: ").strip()
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return

    # Get separation mode
    mode = input("Mode (extract/remove) [extract]: ").strip() or "extract"

    # Get model size
    model_size = input("Model size (small/medium/large) [small]: ").strip() or "small"

    # Get chunk duration
    chunk_duration_str = input("Chunk duration in seconds [25]: ").strip() or "25"
    try:
        chunk_duration = float(chunk_duration_str)
        chunk_duration = max(5.0, min(60.0, chunk_duration))
    except ValueError:
        chunk_duration = 25.0

    print(f"\nUsing chunk duration: {chunk_duration}s")

    # Process prompts in loop
    while True:
        prompt = input("\nEnter text prompt (or 'quit' to exit): ").strip()
        if prompt.lower() in ['quit', 'exit', 'q']:
            break

        output_dir = input("Output directory [output]: ").strip() or "output"

        try:
            separate_audio(
                input_path=input_path,
                prompt=prompt,
                output_dir=output_dir,
                mode=mode,
                model_size=model_size,
                chunk_duration=chunk_duration,
                token=token,
            )
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="SAM Audio CLI - Text-prompted audio separation")

    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input audio file path"
    )

    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Text prompt describing target sound"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )

    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["extract", "remove"],
        default="extract",
        help="Separation mode: extract or remove (default: extract)"
    )

    parser.add_argument(
        "--model-size",
        type=str,
        choices=["small", "medium", "large"],
        default="small",
        help="Model size (default: small)"
    )

    parser.add_argument(
        "--float32",
        action="store_true",
        help="Use float32 precision (slower, more memory)"
    )

    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=25.0,
        help="Audio chunk duration in seconds (5-60) (default: 25.0)"
    )

    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace access token (optional)"
    )

    args = parser.parse_args()

    # If no input provided, run interactive mode
    if not args.input:
        interactive_mode(token=args.token)
        return

    # Validate required args for non-interactive mode
    if not args.prompt:
        print("Error: --prompt is required when using --input")
        return

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    # Run separation
    try:
        separate_audio(
            input_path=args.input,
            prompt=args.prompt,
            output_dir=args.output_dir,
            mode=args.mode,
            model_size=args.model_size,
            use_float32=args.float32,
            chunk_duration=args.chunk_duration,
            token=args.token,
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
