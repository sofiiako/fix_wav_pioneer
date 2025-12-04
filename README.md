# WAV File Fixer for Pioneer DJ Players

A Python script that fixes WAV files to ensure reliable playback on Pioneer DJ hardware (e.g., CDJ players), even if Rekordbox accepts them.

## Problems Fixed

1. **Invalid WAV Header Audio Format Bytes**
   - Detects and corrects invalid AudioFormat values (e.g., `0xEFFF` → `0x0001` for PCM)
   - Properly parses RIFF structure to locate and patch the fmt chunk
   - Ensures all header fields are consistent

2. **Unsupported Sample Rates**
   - Resamples audio to 44,100 Hz if at different sample rates (e.g., 96 kHz)
   - Uses high-quality resampling (scipy.signal.resample)
   - Preserves audio quality and duration

3. **Problematic Bit Depths**
   - Converts 32-bit audio (int or float) to 16-bit PCM with proper scaling and clipping
   - Preserves 16-bit and 24-bit audio as-is
   - Ensures proper dynamic range and prevents distortion

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Install Dependencies
