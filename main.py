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
    """Show GPU memory stats"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[GPU Memory{' - ' + label if label else ''}] "
              f"Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Peak: {max_allocated:.2f}GB")


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
    
    print(f"\n{'='*60}")
    print(f"Loading {model_name}...")
    print(f"{'='*60}")
    
    
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
    
    # Replace _get_video_features to not use vision_encoder
    def _get_video_features_lite(self, video, audio_features):
        B, T, _ = audio_features.shape
        # Always return zeros since we're not using video
        return audio_features.new_zeros(B, self._vision_encoder_dim, T)
    
    # Bind the new method
    import types
    model._get_video_features = types.MethodType(_get_video_features_lite, model)
    
    # Delete rankers
    if hasattr(model, 'visual_ranker') and model.visual_ranker is not None:
        del model.visual_ranker
        model.visual_ranker = None
        gc.collect()
        print("  ✓ Removed visual_ranker")
    
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


def separate_audio(
    input_path: str,
    prompt: str,
    output_dir: str = "output",
    mode: str = "extract",
    model_size: str = "small",
    use_float32: bool = False,
    chunk_duration: float = 25.0,
):
    """
    Separate audio using text prompt
    
    Args:
        input_path: Path to input audio file
        prompt: Text description of sound to extract/remove
        output_dir: Output directory for results
        mode: "extract" or "remove"
        model_size: Model size (small/base/large)
        use_float32: Use float32 precision (better quality, more VRAM)
        chunk_duration: Duration of audio chunks in seconds
    """
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32 if use_float32 or device == 'cpu' else torch.bfloat16
    
    print(f"\n{'='*60}")
    print(f"SAM Audio CLI - Audio Separation")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Precision: {'float32 (High Quality)' if dtype == torch.float32 else 'bfloat16 (Memory Optimized)'}")
    print(f"Model: sam-audio-{model_size}")
    print(f"Input: {input_path}")
    print(f"Prompt: '{prompt}'")
    print(f"Mode: {mode}")
    print(f"{'='*60}\n")
    
    # Validate input file
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Clear GPU memory and reset peak stats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
    
    show_gpu_memory("Before loading model")
    
    # Start timing
    start_time = time.time()
    
    # Create lite model
    model_name = f"facebook/sam-audio-{model_size}"
    model, processor = create_lite_model(model_name)
    model = model.eval().to(device, dtype)
    
    show_gpu_memory("After loading model")
    
    print(f"\nLoading audio: {input_path}")
    
    # Load audio
    sample_rate = processor.audio_sampling_rate
    audio, orig_sr = torchaudio.load(input_path)
    
    if orig_sr != sample_rate:
        print(f"Resampling from {orig_sr}Hz to {sample_rate}Hz...")
        resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
        audio = resampler(audio)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        print("Converting stereo to mono...")
        audio = audio.mean(dim=0, keepdim=True)
    
    # Calculate audio duration
    audio_duration = audio.shape[1] / sample_rate
    print(f"Audio duration: {audio_duration:.2f}s ({audio_duration/60:.2f} min)")
    
    # Chunking settings
    CHUNK_DURATION = max(5.0, min(60.0, chunk_duration))
    MAX_CHUNK_SAMPLES = int(sample_rate * CHUNK_DURATION)
    
    # Process audio (with or without chunking)
    if audio.shape[1] > MAX_CHUNK_SAMPLES:
        print(f"\n{'='*60}")
        print(f"Using chunking mode ({CHUNK_DURATION}s chunks)")
        print(f"{'='*60}\n")
        
        # Split audio into chunks
        audio_tensor = audio.squeeze(0).to(device, dtype)
        chunks = torch.split(audio_tensor, MAX_CHUNK_SAMPLES, dim=-1)
        total_chunks = len(chunks)
        
        print(f"Processing {total_chunks} chunks...")
        
        out_target = []
        out_residual = []
        
        show_gpu_memory("Before chunked separation")
        
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1}/{total_chunks}...", end=" ", flush=True)
            
            # Skip very short chunks
            if chunk.shape[-1] < sample_rate:  # Less than 1 second
                print("(skipped - too short)")
                continue
            
            # Prepare batch for this chunk
            batch = processor(
                audios=[chunk.unsqueeze(0)],
                descriptions=[prompt]
            ).to(device)
            
            # Run separation
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    result = model.separate(
                        batch,
                        predict_spans=False,
                        reranking_candidates=1
                    )
            
            out_target.append(result.target[0].cpu())
            out_residual.append(result.residual[0].cpu())
            
            print("✓")
            
            # Clean up chunk results
            del batch, result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        show_gpu_memory("After chunked separation")
        
        # Concatenate all chunks
        print("\nCombining chunks...")
        target_audio = torch.cat(out_target, dim=-1).clamp(-1, 1).float().unsqueeze(0)
        residual_audio = torch.cat(out_residual, dim=-1).clamp(-1, 1).float().unsqueeze(0)
        
        del out_target, out_residual, chunks, audio_tensor
        
    else:
        print(f"\n{'='*60}")
        print(f"Processing as single batch (no chunking needed)")
        print(f"{'='*60}\n")
        
        # Prepare inputs
        print("Preparing batch...")
        batch = processor(
            audios=[input_path],
            descriptions=[prompt],
        ).to(device)
        
        # Run separation
        print("Running separation...")
        show_gpu_memory("Before separation")
        
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=dtype):
            result = model.separate(
                batch, 
                predict_spans=False,
                reranking_candidates=1
            )
        
        show_gpu_memory("After separation")
        
        target_audio = result.target[0].float().unsqueeze(0).cpu()
        residual_audio = result.residual[0].float().unsqueeze(0).cpu()
        
        del batch, result
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}\n")
    
    # Generate output filenames
    input_stem = input_file.stem
    
    if mode == "extract":
        extracted_path = output_path / f"{input_stem}_extracted.wav"
        residual_path = output_path / f"{input_stem}_residual.wav"
        
        torchaudio.save(str(extracted_path), target_audio, sample_rate)
        torchaudio.save(str(residual_path), residual_audio, sample_rate)
        
        print(f"✓ Extracted audio: {extracted_path}")
        print(f"✓ Residual audio:  {residual_path}")
    else:  # remove
        cleaned_path = output_path / f"{input_stem}_cleaned.wav"
        removed_path = output_path / f"{input_stem}_removed.wav"
        
        torchaudio.save(str(cleaned_path), residual_audio, sample_rate)
        torchaudio.save(str(removed_path), target_audio, sample_rate)
        
        print(f"✓ Cleaned audio:  {cleaned_path}")
        print(f"✓ Removed audio:  {removed_path}")
    
    # Cleanup
    del target_audio, residual_audio, audio
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    show_gpu_memory("After cleanup")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Audio duration:   {audio_duration:.2f}s ({audio_duration/60:.2f} min)")
    print(f"Processing time:  {processing_time:.2f}s ({processing_time/60:.2f} min)")
    print(f"Speed:            {audio_duration/processing_time:.2f}x realtime")
    print(f"Model:            sam-audio-{model_size}")
    print(f"Precision:        {'float32' if dtype == torch.float32 else 'bfloat16'}")
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU Memory:  {peak_gb:.2f}GB")
    print(f"{'='*60}\n")


def interactive_mode():
    """Run in interactive mode with text prompts"""
    print(f"\n{'='*60}")
    print("SAM Audio CLI - Interactive Mode")
    print(f"{'='*60}\n")
    
    # Get input file
    while True:
        input_path = input("Enter audio file path [./audio.mp3]: ").strip().strip('"').strip("'")
        if not input_path:
            input_path = "./audio.mp3"
        if Path(input_path).exists():
            break
        print(f"Error: File not found. Please try again.")
    
    # Get prompt
    prompt = input("Enter description (e.g., 'vocals', 'drums', 'dog barking'): ").strip()
    
    # Get mode
    while True:
        mode = input("Mode (extract/remove) [extract]: ").strip().lower() or "extract"
        if mode in ["extract", "remove"]:
            break
        print("Error: Mode must be 'extract' or 'remove'")
    
    # Get model size
    while True:
        model_size = input("Model size (small/base/large) [small]: ").strip().lower() or "small"
        if model_size in ["small", "base", "large"]:
            break
        print("Error: Model size must be 'small', 'base', or 'large'")
    
    # Get output directory
    output_dir = input("Output directory [output]: ").strip() or "output"
    

    
    # Run separation
    separate_audio(
        input_path=input_path,
        prompt=prompt,
        output_dir=output_dir,
        mode=mode,
        model_size=model_size
    )


def main():
    parser = argparse.ArgumentParser(
        description="SAM Audio CLI - Text-prompted audio separation with memory optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract vocals from a song
  python sam_audio_cli.py --input song.mp3 --prompt "vocals" --mode extract
  
  # Remove drums from a track
  python sam_audio_cli.py --input track.wav --prompt "drums" --mode remove --model base
  
  # Interactive mode
  python sam_audio_cli.py
  
  # High quality mode (uses more VRAM)
  python sam_audio_cli.py --input audio.mp3 --prompt "guitar" --float32
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input audio file path"
    )
    
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Text description of sound to extract/remove (e.g., 'vocals', 'drums', 'dog barking')"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["extract", "remove"],
        default="extract",
        help="Mode: 'extract' to isolate sound, 'remove' to eliminate it (default: extract)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["small", "base", "large"],
        default="small",
        help="Model size: small (~6GB VRAM), base (~7GB), large (~10GB) (default: small)"
    )
    
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Use float32 precision for better quality (uses more VRAM)"
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
        interactive_mode()
    else:
        # Validate required arguments
        if not args.prompt:
            parser.error("--prompt is required when using --input")
        
        # Run separation
        separate_audio(
            input_path=args.input,
            prompt=args.prompt,
            output_dir=args.output,
            mode=args.mode,
            model_size=args.model,
            use_float32=args.float32,
            chunk_duration=args.chunk_duration
        )


if __name__ == "__main__":
    main()