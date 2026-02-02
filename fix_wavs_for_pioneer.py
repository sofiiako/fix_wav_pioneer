#!/usr/bin/env python3
"""
WAV File Fixer for Pioneer DJ Players

This script scans a folder of WAV files and fixes them to ensure reliable playback
on Pioneer DJ hardware (e.g., CDJ players), even if Rekordbox accepts them.

Fixes applied:
1. Corrects invalid AudioFormat bytes in WAV headers (e.g., 0xEFFF -> 0x0001 for PCM)
2. Resamples audio to 44,100 Hz if at different sample rates
3. Converts 32-bit audio to 16-bit PCM (preserves 16-bit and 24-bit)

Dependencies:
    pip install numpy soundfile scipy

Usage:
    python fix_wavs_for_pioneer.py /path/to/input_dir --output-dir /path/to/output_dir
    python fix_wavs_for_pioneer.py /path/to/input_dir --in-place
"""

import argparse
import logging
import os
import struct
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import soundfile as sf
from scipy import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
TARGET_SAMPLE_RATE = 44100
TARGET_BIT_DEPTH_32BIT = 16  # Convert 32-bit to 16-bit
SUPPORTED_BIT_DEPTHS = {16, 24}  # These are kept as-is
PCM_FORMAT = 0x0001
WAVE_FORMAT_EXTENSIBLE = 0xFFFE


class WAVHeader:
    """Represents a WAV file header with all relevant fields."""

    def __init__(self):
        self.riff_chunk_size = 0
        self.fmt_chunk_offset = 0
        self.fmt_chunk_size = 0
        self.audio_format = 0
        self.num_channels = 0
        self.sample_rate = 0
        self.byte_rate = 0
        self.block_align = 0
        self.bits_per_sample = 0
        self.data_chunk_offset = 0
        self.data_chunk_size = 0
        self.extra_data = {}  # Store any other chunks


def parse_wav_header(file_path: Path) -> Optional[Tuple[WAVHeader, bytes]]:
    """
    Parse WAV file header and return header info and raw file data.

    Args:
        file_path: Path to the WAV file

    Returns:
        Tuple of (WAVHeader object, raw file bytes) or None if parsing fails
    """
    try:
        with open(file_path, 'rb') as f:
            data = f.read()

        # Check RIFF header
        if len(data) < 12 or data[0:4] != b'RIFF' or data[8:12] != b'WAVE':
            logger.error(f"{file_path}: Not a valid WAV file (invalid RIFF/WAVE header)")
            return None

        header = WAVHeader()
        header.riff_chunk_size = struct.unpack('<I', data[4:8])[0]

        # Parse chunks
        pos = 12
        fmt_found = False
        data_found = False

        while pos < len(data) - 8:
            chunk_id = data[pos:pos+4]
            chunk_size = struct.unpack('<I', data[pos+4:pos+8])[0]

            if chunk_id == b'fmt ':
                fmt_found = True
                header.fmt_chunk_offset = pos
                header.fmt_chunk_size = chunk_size

                if chunk_size < 16:
                    logger.error(f"{file_path}: Invalid fmt chunk size")
                    return None

                fmt_data = data[pos+8:pos+8+chunk_size]
                header.audio_format = struct.unpack('<H', fmt_data[0:2])[0]
                header.num_channels = struct.unpack('<H', fmt_data[2:4])[0]
                header.sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
                header.byte_rate = struct.unpack('<I', fmt_data[8:12])[0]
                header.block_align = struct.unpack('<H', fmt_data[12:14])[0]
                header.bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]

                pos += 8 + chunk_size

            elif chunk_id == b'data':
                data_found = True
                header.data_chunk_offset = pos
                header.data_chunk_size = chunk_size
                pos += 8 + chunk_size

            else:
                # Store other chunks
                header.extra_data[chunk_id] = data[pos:pos+8+chunk_size]
                pos += 8 + chunk_size

            # Ensure even alignment
            if chunk_size % 2 == 1:
                pos += 1

        if not fmt_found or not data_found:
            logger.error(f"{file_path}: Missing fmt or data chunk")
            return None

        return header, data

    except Exception as e:
        logger.error(f"{file_path}: Error parsing WAV header: {e}")
        return None


def needs_fixing(header: WAVHeader) -> Dict[str, Any]:
    """
    Determine what fixes are needed for this WAV file.

    Args:
        header: Parsed WAV header

    Returns:
        Dictionary with fix requirements
    """
    fixes = {
        'header_fix': False,
        'resample': False,
        'bit_depth_conversion': False,
        'original_format': header.audio_format,
        'original_sample_rate': header.sample_rate,
        'original_bit_depth': header.bits_per_sample,
        'target_sample_rate': TARGET_SAMPLE_RATE,
        'target_bit_depth': header.bits_per_sample
    }

    # Check audio format
    if header.audio_format not in [PCM_FORMAT, WAVE_FORMAT_EXTENSIBLE]:
        logger.info(f"Audio format 0x{header.audio_format:04X} needs fixing to PCM (0x0001)")
        fixes['header_fix'] = True
    elif header.audio_format == WAVE_FORMAT_EXTENSIBLE:
        # WAVE_FORMAT_EXTENSIBLE might be OK, but we'll normalize to PCM if data is PCM
        logger.info(f"WAVE_FORMAT_EXTENSIBLE detected, will normalize to PCM if appropriate")
        fixes['header_fix'] = True

    # Special case: check for the known bad pattern (0xEFFF)
    if header.audio_format == 0xEFFF:
        logger.info(f"Detected known bad format 0xEFFF, will fix to 0x0001")
        fixes['header_fix'] = True

    # Check sample rate
    if header.sample_rate != TARGET_SAMPLE_RATE:
        logger.info(f"Sample rate {header.sample_rate} Hz needs resampling to {TARGET_SAMPLE_RATE} Hz")
        fixes['resample'] = True

    # Check bit depth
    if header.bits_per_sample == 32:
        logger.info(f"32-bit audio needs conversion to 16-bit")
        fixes['bit_depth_conversion'] = True
        fixes['target_bit_depth'] = TARGET_BIT_DEPTH_32BIT
    elif header.bits_per_sample not in SUPPORTED_BIT_DEPTHS:
        logger.warning(f"Unusual bit depth {header.bits_per_sample}, will attempt to normalize")
        fixes['bit_depth_conversion'] = True
        fixes['target_bit_depth'] = 16

    return fixes


def load_audio_data(file_path: Path, header: WAVHeader, raw_data: bytes) -> Optional[Tuple[np.ndarray, int]]:
    """
    Load audio data from WAV file.

    Args:
        file_path: Path to WAV file
        header: Parsed header
        raw_data: Raw file bytes

    Returns:
        Tuple of (audio data as numpy array, sample rate) or None if loading fails
    """
    try:
        # Use soundfile to load the audio - it handles various formats well
        audio_data, sample_rate = sf.read(str(file_path), dtype='float32')
        return audio_data, sample_rate

    except Exception as e:
        logger.error(f"{file_path}: Error loading audio data with soundfile: {e}")

        # Fallback: try to parse raw PCM data from the data chunk
        try:
            data_offset = header.data_chunk_offset + 8
            data_size = header.data_chunk_size
            pcm_data = raw_data[data_offset:data_offset + data_size]

            # Determine dtype based on bit depth
            if header.bits_per_sample == 16:
                dtype = np.int16
            elif header.bits_per_sample == 24:
                # 24-bit is tricky, convert to int32
                num_samples = len(pcm_data) // 3 // header.num_channels
                audio_data = np.zeros((num_samples, header.num_channels), dtype=np.int32)
                for i in range(num_samples * header.num_channels):
                    sample_bytes = pcm_data[i*3:(i+1)*3]
                    # Sign extend 24-bit to 32-bit
                    value = int.from_bytes(sample_bytes, byteorder='little', signed=False)
                    if value & 0x800000:  # Check sign bit
                        value |= 0xFF000000
                    else:
                        value &= 0x00FFFFFF
                    sample_idx = i // header.num_channels
                    channel_idx = i % header.num_channels
                    audio_data[sample_idx, channel_idx] = value
                # Normalize to float32
                audio_data = audio_data.astype(np.float32) / (2**23)
                if header.num_channels == 1:
                    audio_data = audio_data.flatten()
                return audio_data, header.sample_rate
            elif header.bits_per_sample == 32:
                dtype = np.int32
            else:
                logger.error(f"{file_path}: Unsupported bit depth {header.bits_per_sample}")
                return None

            if header.bits_per_sample != 24:
                audio_data = np.frombuffer(pcm_data, dtype=dtype)

                # Reshape for multi-channel
                if header.num_channels > 1:
                    audio_data = audio_data.reshape(-1, header.num_channels)

                # Normalize to float32 range [-1.0, 1.0]
                if dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0

                return audio_data, header.sample_rate

        except Exception as e2:
            logger.error(f"{file_path}: Fallback audio loading also failed: {e2}")
            return None


def resample_audio(audio_data: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    """
    Resample audio data to target sample rate using high-quality resampling.

    Args:
        audio_data: Audio data as numpy array (samples x channels or just samples for mono)
        original_rate: Original sample rate
        target_rate: Target sample rate

    Returns:
        Resampled audio data
    """
    if original_rate == target_rate:
        return audio_data

    logger.info(f"Resampling from {original_rate} Hz to {target_rate} Hz")

    # Calculate the number of output samples
    num_samples = audio_data.shape[0]
    num_output_samples = int(num_samples * target_rate / original_rate)

    # Handle mono vs stereo
    if audio_data.ndim == 1:
        # Mono
        resampled = signal.resample(audio_data, num_output_samples)
    else:
        # Multi-channel: resample each channel separately
        num_channels = audio_data.shape[1]
        resampled = np.zeros((num_output_samples, num_channels), dtype=np.float32)
        for ch in range(num_channels):
            resampled[:, ch] = signal.resample(audio_data[:, ch], num_output_samples)

    return resampled.astype(np.float32)


def convert_bit_depth(audio_data: np.ndarray, target_bit_depth: int) -> np.ndarray:
    """
    Convert audio data to target bit depth with proper scaling and clipping.

    Args:
        audio_data: Audio data as float32 in range [-1.0, 1.0]
        target_bit_depth: Target bit depth (16 or 24)

    Returns:
        Audio data as float32 (soundfile will handle final conversion)
    """
    # Ensure audio data is in float32 range [-1.0, 1.0]
    # Clip to prevent overflow
    audio_data = np.clip(audio_data, -1.0, 1.0)

    logger.info(f"Converting to {target_bit_depth}-bit PCM")

    # Return as float32 - soundfile will handle the conversion when writing
    return audio_data.astype(np.float32)


def write_fixed_wav(output_path: Path, audio_data: np.ndarray, sample_rate: int, 
                    bit_depth: int, num_channels: int) -> bool:
    """
    Write fixed WAV file with proper headers.

    Args:
        output_path: Output file path
        audio_data: Audio data as numpy array (float32, range [-1.0, 1.0])
        sample_rate: Sample rate
        bit_depth: Bits per sample (16 or 24)
        num_channels: Number of channels

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine the subtype for soundfile
        if bit_depth == 16:
            subtype = 'PCM_16'
        elif bit_depth == 24:
            subtype = 'PCM_24'
        else:
            logger.error(f"Unsupported target bit depth: {bit_depth}")
            return False

        # Write the file with soundfile (it creates proper headers)
        sf.write(
            str(output_path),
            audio_data,
            sample_rate,
            subtype=subtype,
            format='WAV'
        )

        logger.info(f"Successfully wrote fixed WAV: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error writing WAV file {output_path}: {e}")
        return False


def fix_wav_file(input_path: Path, output_path: Path) -> bool:
    """
    Fix a single WAV file for Pioneer player compatibility.

    Args:
        input_path: Input WAV file path
        output_path: Output WAV file path

    Returns:
        True if successful or no fixes needed, False on error
    """
    logger.info(f"Processing: {input_path}")

    # Parse header
    result = parse_wav_header(input_path)
    if result is None:
        return False

    header, raw_data = result

    # Log original file info
    logger.info(f"  Format: 0x{header.audio_format:04X}, "
                f"Sample Rate: {header.sample_rate} Hz, "
                f"Bit Depth: {header.bits_per_sample}, "
                f"Channels: {header.num_channels}")

    # Determine what fixes are needed
    fixes = needs_fixing(header)

    if not any([fixes['header_fix'], fixes['resample'], fixes['bit_depth_conversion']]):
        logger.info(f"  No fixes needed, file is already compatible")
        # Copy file if output path is different
        if input_path != output_path:
            import shutil
            shutil.copy2(input_path, output_path)
        return True

    # Load audio data
    audio_result = load_audio_data(input_path, header, raw_data)
    if audio_result is None:
        return False

    audio_data, sample_rate = audio_result

    # Apply fixes
    fixed_audio = audio_data
    fixed_sample_rate = sample_rate
    fixed_bit_depth = header.bits_per_sample

    # Resample if needed
    if fixes['resample']:
        fixed_audio = resample_audio(fixed_audio, sample_rate, TARGET_SAMPLE_RATE)
        fixed_sample_rate = TARGET_SAMPLE_RATE

    # Convert bit depth if needed
    if fixes['bit_depth_conversion']:
        fixed_audio = convert_bit_depth(fixed_audio, fixes['target_bit_depth'])
        fixed_bit_depth = fixes['target_bit_depth']

    # Write fixed file
    # If only header fix is needed and no audio processing, we could do a binary patch
    # But for simplicity and correctness, we'll always rewrite the full file
    success = write_fixed_wav(
        output_path,
        fixed_audio,
        fixed_sample_rate,
        fixed_bit_depth,
        header.num_channels
    )

    if success:
        logger.info(f"  Fixed: {output_path}")
        logger.info(f"  New specs: PCM 0x0001, {fixed_sample_rate} Hz, "
                   f"{fixed_bit_depth}-bit, {header.num_channels} channels")

    return success


def process_directory(input_dir: Path, output_dir: Optional[Path], in_place: bool) -> Tuple[int, int]:
    """
    Process all WAV files in a directory recursively.

    Args:
        input_dir: Input directory path
        output_dir: Output directory path (None if in_place)
        in_place: Whether to modify files in place

    Returns:
        Tuple of (successful_count, failed_count)
    """
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 0, 0

    # Find all WAV files
    wav_files = list(input_dir.rglob('*.wav')) + list(input_dir.rglob('*.WAV'))

    if not wav_files:
        logger.warning(f"No WAV files found in {input_dir}")
        return 0, 0

    logger.info(f"Found {len(wav_files)} WAV file(s) to process")

    successful = 0
    failed = 0

    for wav_file in wav_files:
        try:
            if in_place:
                # Create a temporary file, then replace original
                temp_output = wav_file.with_suffix('.wav.tmp')
                if fix_wav_file(wav_file, temp_output):
                    temp_output.replace(wav_file)
                    successful += 1
                else:
                    failed += 1
                    if temp_output.exists():
                        temp_output.unlink()
            else:
                # Preserve directory structure in output
                rel_path = wav_file.relative_to(input_dir)
                output_path = output_dir / rel_path

                if fix_wav_file(wav_file, output_path):
                    successful += 1
                else:
                    failed += 1

        except Exception as e:
            logger.error(f"Unexpected error processing {wav_file}: {e}")
            failed += 1

    return successful, failed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fix WAV files for reliable playback on Pioneer DJ players',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix a single file with custom output name
  python fix_wavs_for_pioneer.py input.wav -o output_fixed.wav

  # Fix a single file in place (overwrites original)
  python fix_wavs_for_pioneer.py input.wav --in-place

  # Fix directory and save to output directory
  python fix_wavs_for_pioneer.py /path/to/wavs --output-dir /path/to/fixed_wavs

  # Fix directory files in place (overwrites originals)
  python fix_wavs_for_pioneer.py /path/to/wavs --in-place

  # Enable verbose logging
  python fix_wavs_for_pioneer.py /path/to/wavs --output-dir /path/to/fixed_wavs -v
        """
    )

    parser.add_argument(
        'input_path',
        type=str,
        help='Input WAV file or directory containing WAV files'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for fixed WAV files (required for directory processing unless --in-place is used)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file path for single file conversion (only valid when input is a single file)'
    )

    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Modify files in place (overwrites originals)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = Path(args.input_path).resolve()

    # Check if input is a file or directory
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")

    is_single_file = input_path.is_file()

    # Validate arguments based on mode
    if is_single_file:
        # Single file mode
        if not args.output and not args.in_place:
            parser.error("When processing a single file, either --output (-o) or --in-place must be specified")
        if args.in_place and args.output:
            parser.error("Cannot use both --in-place and --output for a single file")
        if args.output_dir:
            parser.error("Cannot use --output-dir with single file mode, use --output instead")

        # Process single file
        logger.info("=" * 60)
        logger.info("WAV File Fixer for Pioneer DJ Players")
        logger.info("=" * 60)
        logger.info(f"Input file: {input_path}")

        if args.in_place:
            logger.info("Mode: In-place (will overwrite original)")
            logger.info("=" * 60)

            # Create a temporary file, then replace original
            temp_output = input_path.with_suffix('.wav.tmp')
            success = fix_wav_file(input_path, temp_output)

            if success:
                temp_output.replace(input_path)
            else:
                if temp_output.exists():
                    temp_output.unlink()
        else:
            output_path = Path(args.output).resolve()

            # Prevent overwriting input file
            if output_path == input_path:
                parser.error("Output file cannot be the same as input file. Use --in-place if you want to overwrite.")

            logger.info(f"Output file: {output_path}")
            logger.info("=" * 60)

            success = fix_wav_file(input_path, output_path)

        logger.info("=" * 60)
        if success:
            logger.info("Processing complete! File fixed successfully.")
        else:
            logger.error("Processing failed!")
        logger.info("=" * 60)

        sys.exit(0 if success else 1)

    else:
        # Directory mode
        if args.output:
            parser.error("--output (-o) can only be used with a single file, not a directory. Use --output-dir for directories.")

        if not args.in_place and not args.output_dir:
            parser.error("For directory processing, either --output-dir or --in-place must be specified")

        if args.in_place and args.output_dir:
            parser.error("Cannot use both --in-place and --output-dir")

        input_dir = input_path
        output_dir = Path(args.output_dir).resolve() if args.output_dir else None

        # Process files
        logger.info("=" * 60)
        logger.info("WAV File Fixer for Pioneer DJ Players")
        logger.info("=" * 60)
        logger.info(f"Input directory: {input_dir}")
        if args.in_place:
            logger.info("Mode: In-place (will overwrite originals)")
        else:
            logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 60)

        successful, failed = process_directory(input_dir, output_dir, args.in_place)

        # Summary
        logger.info("=" * 60)
        logger.info("Processing complete!")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info("=" * 60)

        # Exit with appropriate code
        sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
