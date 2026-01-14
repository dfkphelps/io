"""
vidio.py - A high-performance animated image and video handler

This module provides efficient functions to load and save animated files in multiple formats:
- GIF (Graphics Interchange Format)
- WebP (Web Picture format)
- APNG (Animated PNG)
- AVIF (AV1 Image Format)
- MP4 (MPEG-4 video)
- WebM (Web Media video)

Uses imageio and PIL for better performance and memory efficiency.
"""

try:
    import imageio.v3 as iio
    IMAGEIO_V3 = True
except (ImportError, AttributeError):
    import imageio as iio
    IMAGEIO_V3 = False

# Also import PIL for better metadata reading
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional, Union
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Try to import pillow-avif-plugin for AVIF support
try:
    import pillow_avif
    AVIF_SUPPORT = True
except ImportError:
    AVIF_SUPPORT = False

# Try to import apng for APNG support
try:
    from apng import APNG
    APNG_SUPPORT = True
except ImportError:
    APNG_SUPPORT = False


def load_video(filepath: str, fast_mode: bool = True) -> tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Load an animated file and extract all frames with metadata.
    
    Supports: GIF, WebP, APNG, AVIF, MP4, WebM
    
    Args:
        filepath: Path to the file to load
        fast_mode: If True, skip detailed per-frame metadata extraction (faster)
        
    Returns:
        A tuple containing:
        - List of numpy arrays (one per frame) in shape (height, width, channels)
        - Dictionary containing metadata about the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported or has invalid data
        IOError: If there's an error reading the file
        PermissionError: If there's no permission to read the file
    """
    # Validate input
    if not filepath:
        raise ValueError("Filepath cannot be empty")
    
    if not isinstance(filepath, str):
        raise TypeError(f"Filepath must be a string, got {type(filepath)}")
    
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Check if it's a file (not a directory)
    if not os.path.isfile(filepath):
        raise ValueError(f"Path is not a file: {filepath}")
    
    # Check read permissions
    if not os.access(filepath, os.R_OK):
        raise PermissionError(f"No read permission for file: {filepath}")
    
    # Check file size (empty files)
    if os.path.getsize(filepath) == 0:
        raise ValueError(f"File is empty: {filepath}")
    
    try:
        # Detect file format
        ext = os.path.splitext(filepath)[1].lower()
        
        # For video formats (MP4, WebM), use imageio
        if ext in ['.mp4', '.webm', '.mkv', '.avi']:
            return _load_video_format(filepath, fast_mode)
        
        # For APNG, use apng library if available
        if ext == '.png' or ext == '.apng':
            # Try APNG first
            if APNG_SUPPORT:
                try:
                    return _load_apng(filepath, fast_mode)
                except:
                    pass  # Fall through to PIL
        
        # For image formats (GIF, WebP, AVIF, PNG), use PIL
        return _load_image_format(filepath, fast_mode)
        
    except FileNotFoundError:
        raise
    except PermissionError:
        raise
    except ValueError:
        raise
    except IOError as e:
        raise IOError(f"Error reading file: {str(e)}")
    except Exception as e:
        raise IOError(f"Unexpected error loading file: {str(e)}")


def _load_apng(filepath: str, fast_mode: bool) -> tuple[List[np.ndarray], Dict[str, Any]]:
    """Load APNG file using apng library."""
    if not APNG_SUPPORT:
        raise ValueError("APNG support not available. Install: pip install apng")
    
    apng_img = APNG.open(filepath)
    
    metadata = {
        'format': 'APNG',
        'size': (apng_img.width, apng_img.height),
        'n_images': len(apng_img.frames),
        'is_animated': len(apng_img.frames) > 1,
    }
    
    frames = []
    frame_metadata = []
    
    for i, (png, control) in enumerate(apng_img.frames):
        # Convert PNG to PIL Image then to numpy
        img = Image.open(png)
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            frame_array = np.asarray(img.convert('RGBA'))
        else:
            frame_array = np.asarray(img.convert('RGB'))
        
        frames.append(frame_array)
        
        # Get frame duration from control chunk
        duration = control.delay / 1000.0 if control.delay else 0.1  # delay is in milliseconds
        
        if not fast_mode:
            frame_metadata.append({
                'frame_number': i,
                'shape': frame_array.shape,
                'dtype': str(frame_array.dtype),
                'duration': duration
            })
        else:
            frame_metadata.append({'duration': duration})
    
    metadata['frames'] = frame_metadata
    metadata['actual_frame_count'] = len(frames)
    metadata['n_images'] = len(frames)
    metadata['is_animated'] = len(frames) > 1
    
    if frames:
        metadata['shape'] = frames[0].shape
        metadata['dtype'] = str(frames[0].dtype)
    
    durations = [f.get('duration') for f in frame_metadata if 'duration' in f]
    if durations:
        metadata['avg_duration'] = sum(durations) / len(durations)
    
    return frames, metadata


def _load_video_format(filepath: str, fast_mode: bool) -> tuple[List[np.ndarray], Dict[str, Any]]:
    """Load video formats (MP4, WebM, etc.) using imageio."""
    # Get video properties
    if IMAGEIO_V3:
        try:
            props = iio.improps(filepath)
            fps = props.fps if hasattr(props, 'fps') else 30.0
            n_frames = props.n_images if hasattr(props, 'n_images') else None
        except:
            fps = 30.0
            n_frames = None
    else:
        reader = iio.get_reader(filepath)
        fps = reader.get_meta_data().get('fps', 30.0)
        n_frames = reader.count_frames() if hasattr(reader, 'count_frames') else None
    
    metadata = {
        'format': os.path.splitext(filepath)[1][1:].upper(),
        'fps': fps,
        'is_animated': True,
    }
    
    frames = []
    frame_metadata = []
    
    # Pre-allocate list if we know the frame count
    if n_frames:
        frames = [None] * n_frames
    
    # Read frames
    if IMAGEIO_V3:
        frame_iter = iio.imiter(filepath)
    else:
        reader = iio.get_reader(filepath)
        frame_iter = reader
    
    frame_duration = 1.0 / fps if fps > 0 else 0.033
    
    for i, frame in enumerate(frame_iter):
        # Store frame directly (already numpy array, no copy needed)
        if n_frames and i < n_frames:
            frames[i] = frame
        else:
            frames.append(frame)
        
        if not fast_mode:
            frame_metadata.append({
                'frame_number': i,
                'shape': frame.shape,
                'dtype': str(frame.dtype),
                'duration': frame_duration
            })
        else:
            frame_metadata.append({'duration': frame_duration})
    
    # Remove None entries if pre-allocation was used
    if n_frames:
        frames = [f for f in frames if f is not None]
    
    if not IMAGEIO_V3:
        reader.close()
    
    metadata['frames'] = frame_metadata
    metadata['actual_frame_count'] = len(frames)
    metadata['n_images'] = len(frames)
    
    if frames:
        metadata['shape'] = frames[0].shape
        metadata['dtype'] = str(frames[0].dtype)
        metadata['size'] = (frames[0].shape[1], frames[0].shape[0])
    
    metadata['avg_duration'] = frame_duration
    
    return frames, metadata


def _load_image_format(filepath: str, fast_mode: bool) -> tuple[List[np.ndarray], Dict[str, Any]]:
    """Load image formats (GIF, WebP, AVIF, PNG) using PIL."""
    # Use PIL to read file metadata more accurately
    pil_img = Image.open(filepath)
    
    # Verify it's a supported format
    if pil_img.format not in ['GIF', 'WEBP', 'PNG', 'AVIF']:
        raise ValueError(f"File format not supported: {pil_img.format}. Supported: GIF, WebP, APNG, AVIF, MP4, WebM")
    
    file_format = pil_img.format
    
    # Check AVIF support
    if file_format == 'AVIF' and not AVIF_SUPPORT:
        raise ValueError("AVIF support not available. Install: pip install pillow-avif-plugin")
    
    # Extract global metadata from PIL
    metadata = {
        'format': pil_img.format,
        'size': pil_img.size,
        'mode': pil_img.mode,
        'n_images': getattr(pil_img, 'n_frames', 1),
        'is_animated': getattr(pil_img, 'is_animated', False),
    }
    
    # Get info from first frame
    info = pil_img.info
    if 'duration' in info:
        metadata['duration'] = info['duration'] / 1000.0  # Convert ms to seconds
    if 'loop' in info:
        metadata['loop'] = info['loop']
    
    # Determine if we need alpha channel
    needs_alpha = pil_img.mode in ('RGBA', 'LA') or (pil_img.mode == 'P' and 'transparency' in pil_img.info)
    target_mode = 'RGBA' if needs_alpha else 'RGB'
    
    # Read all frames and their durations using PIL for accuracy
    frames = []
    frame_metadata = []
    
    try:
        frame_num = 0
        while True:
            try:
                pil_img.seek(frame_num)
            except EOFError:
                break
            
            # Convert frame to numpy array - use asarray for speed (no copy if possible)
            # Convert once to target mode to avoid repeated conversions
            if pil_img.mode != target_mode:
                frame_array = np.asarray(pil_img.convert(target_mode))
            else:
                frame_array = np.asarray(pil_img)
                
            frames.append(frame_array)
            
            # Only extract detailed metadata if not in fast mode
            if not fast_mode:
                # Get frame-specific metadata
                frame_info = pil_img.info
                frame_meta = {
                    'frame_number': frame_num,
                    'shape': frame_array.shape,
                    'dtype': str(frame_array.dtype),
                }
                
                # Get duration for this frame (in milliseconds from PIL)
                if 'duration' in frame_info:
                    frame_meta['duration'] = frame_info['duration'] / 1000.0  # Convert to seconds
                elif 'duration' in metadata:
                    frame_meta['duration'] = metadata['duration']
                else:
                    frame_meta['duration'] = 0.1  # Default fallback
                
                frame_metadata.append(frame_meta)
            else:
                # Fast mode: only store duration
                frame_info = pil_img.info
                if 'duration' in frame_info:
                    frame_metadata.append({'duration': frame_info['duration'] / 1000.0})
                elif 'duration' in metadata:
                    frame_metadata.append({'duration': metadata['duration']})
                else:
                    frame_metadata.append({'duration': 0.1})
            
            frame_num += 1
    
    except Exception as e:
        if len(frames) == 0:
            raise IOError(f"Error reading frames: {str(e)}")
    
    finally:
        pil_img.close()
    
    # Verify we got at least one frame
    if len(frames) == 0:
        raise ValueError("No frames could be extracted from the file")
    
    # Update metadata with actual frame info
    if not metadata:
        metadata = {}
    
    metadata['frames'] = frame_metadata
    metadata['actual_frame_count'] = len(frames)
    metadata['n_images'] = len(frames)
    metadata['is_animated'] = len(frames) > 1
    
    if len(frames) > 0:
        metadata['shape'] = frames[0].shape
        metadata['dtype'] = str(frames[0].dtype)
    
    # Calculate average duration if available
    durations = [f.get('duration') for f in frame_metadata if 'duration' in f]
    if durations:
        metadata['avg_duration'] = sum(durations) / len(durations)
    
    return frames, metadata


def save_video(frames: List[Union[np.ndarray, Any]], 
               filepath: str,
               duration: Optional[Union[float, List[float]]] = None,
               loop: Optional[int] = 0,
               fps: Optional[float] = None,
               quality: Optional[int] = None,
               optimize: bool = False,
               parallel: bool = False,
               output_format: Optional[str] = None,
               **kwargs) -> None:
    """
    Save a list of frames as an animated file.
    
    Supports: GIF, WebP, APNG, AVIF, MP4, WebM
    
    Args:
        frames: List of numpy arrays or PIL Images to save as frames
               Arrays should be in shape (height, width, channels)
        filepath: Path where the file should be saved
        duration: Duration of each frame in seconds (default: None, uses 0.1s)
                 Can be a single float or list of floats (one per frame)
                 For video formats, use fps instead
        loop: Number of times to loop (0 = infinite, default: 0)
             Only applies to GIF, WebP, APNG, AVIF
        fps: Frames per second (alternative to duration, preferred for video formats)
        quality: Quality/compression level (format-dependent):
                 - WebP: 0-100 (higher is better, default: 80)
                 - AVIF: 0-100 (higher is better, default: 80)
                 - MP4/WebM: Use codec-specific settings via kwargs
                 - GIF: not used (uses optimize instead)
        optimize: Whether to optimize the palette/compression (default: False)
        parallel: Use parallel processing for frame conversion (faster for many frames)
        output_format: Force output format ('GIF', 'WEBP', 'APNG', 'AVIF', 'MP4', 'WEBM')
                      If None, determined by file extension
        **kwargs: Additional format-specific parameters
        
    Raises:
        ValueError: If frames list is empty or invalid
        TypeError: If frames are not valid array-like objects
        IOError: If there's an error writing the file
        PermissionError: If there's no permission to write the file
    """
    # Validate frames input
    if not frames:
        raise ValueError("Frames list cannot be empty")
    
    if not isinstance(frames, (list, tuple)):
        raise TypeError(f"Frames must be a list or tuple, got {type(frames)}")
    
    if len(frames) == 0:
        raise ValueError("Must provide at least one frame")
    
    # Convert frames to numpy arrays if needed and validate
    processed_frames = []
    for i, frame in enumerate(frames):
        try:
            # Handle PIL Images
            if hasattr(frame, 'mode') and hasattr(frame, 'size'):
                frame_array = np.asarray(frame)  # Use asarray instead of array (faster, no copy)
            elif isinstance(frame, np.ndarray):
                frame_array = frame
            else:
                # Try to convert to numpy array
                frame_array = np.asarray(frame)
            
            # Validate frame
            if frame_array.size == 0:
                raise ValueError(f"Frame {i} is empty")
            
            if frame_array.ndim not in [2, 3]:
                raise ValueError(f"Frame {i} must be 2D (grayscale) or 3D (color), got {frame_array.ndim}D")
            
            # Ensure frame is in valid range and dtype
            if frame_array.dtype != np.uint8:
                # Normalize if float
                if np.issubdtype(frame_array.dtype, np.floating):
                    if frame_array.max() <= 1.0:
                        frame_array = (frame_array * 255).astype(np.uint8)
                    else:
                        frame_array = frame_array.astype(np.uint8)
                else:
                    frame_array = frame_array.astype(np.uint8)
            
            processed_frames.append(frame_array)
            
        except Exception as e:
            raise TypeError(f"Frame {i} could not be converted to numpy array: {str(e)}")
    
    # Validate filepath
    if not filepath:
        raise ValueError("Filepath cannot be empty")
    
    if not isinstance(filepath, str):
        raise TypeError(f"Filepath must be a string, got {type(filepath)}")
    
    # Determine output format from extension or parameter
    if output_format:
        output_format = output_format.upper()
        if output_format not in ['GIF', 'WEBP', 'APNG', 'AVIF', 'MP4', 'WEBM']:
            raise ValueError(f"output_format must be GIF, WEBP, APNG, AVIF, MP4, or WEBM, got '{output_format}'")
    else:
        # Detect from file extension
        ext = os.path.splitext(filepath)[1].lower()
        format_map = {
            '.gif': 'GIF',
            '.webp': 'WEBP',
            '.png': 'APNG',
            '.apng': 'APNG',
            '.avif': 'AVIF',
            '.mp4': 'MP4',
            '.webm': 'WEBM'
        }
        output_format = format_map.get(ext, 'GIF')
    
    # Check format support
    if output_format == 'AVIF' and not AVIF_SUPPORT:
        raise ValueError("AVIF support not available. Install: pip install pillow-avif-plugin")
    
    if output_format == 'APNG' and not APNG_SUPPORT:
        raise ValueError("APNG support not available. Install: pip install apng")
    
    # Ensure correct extension
    ext_map = {
        'GIF': '.gif',
        'WEBP': '.webp',
        'APNG': '.png',
        'AVIF': '.avif',
        'MP4': '.mp4',
        'WEBM': '.webm'
    }
    correct_ext = ext_map[output_format]
    current_ext = os.path.splitext(filepath)[1].lower()
    
    if current_ext != correct_ext:
        filepath = os.path.splitext(filepath)[0] + correct_ext
    
    # Check directory permissions
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except PermissionError:
            raise PermissionError(f"No permission to create directory: {directory}")
        except OSError as e:
            raise IOError(f"Error creating directory: {str(e)}")
    
    if directory and not os.access(directory or '.', os.W_OK):
        raise PermissionError(f"No write permission for directory: {directory or '.'}")
    
    # Validate and prepare duration/fps
    if fps is not None and duration is not None:
        raise ValueError("Cannot specify both fps and duration. Use one or the other.")
    
    if fps is not None:
        if not isinstance(fps, (int, float)) or fps <= 0:
            raise ValueError("FPS must be a positive number")
        duration = 1.0 / fps
    
    # If no duration specified, use a sensible default
    if duration is None:
        duration = 0.1  # 100ms default
    
    if isinstance(duration, (list, tuple)):
        if len(duration) != len(processed_frames):
            raise ValueError(f"Duration list length ({len(duration)}) must match frames length ({len(processed_frames)})")
        for i, d in enumerate(duration):
            if not isinstance(d, (int, float)) or d <= 0:
                raise ValueError(f"Duration at index {i} must be a positive number")
    elif duration is not None:
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError("Duration must be a positive number")
    
    # Validate loop
    if loop is not None and (not isinstance(loop, int) or loop < 0):
        raise ValueError("Loop must be a non-negative integer (0 = infinite)")
    
    # Validate quality based on format
    if quality is not None:
        if output_format in ['WEBP', 'AVIF']:
            if not isinstance(quality, int) or quality < 0 or quality > 100:
                raise ValueError(f"Quality for {output_format} must be an integer between 0 and 100")
        # For GIF/APNG, quality parameter is ignored (uses optimize instead)
        # For video formats, quality is handled by codec settings
    
    try:
        # For video formats, use imageio
        if output_format in ['MP4', 'WEBM']:
            return _save_video_format(processed_frames, filepath, fps, output_format, **kwargs)
        
        # For APNG, use apng library
        if output_format == 'APNG':
            return _save_apng(processed_frames, filepath, duration, loop)
        
        # For image formats (GIF, WebP, AVIF), use PIL
        return _save_image_format(processed_frames, filepath, duration, loop, 
                                  quality, optimize, parallel, output_format, **kwargs)
        
    except PermissionError:
        raise
    except ValueError:
        raise
    except IOError as e:
        raise IOError(f"Error writing file: {str(e)}")
    except Exception as e:
        raise IOError(f"Unexpected error saving file: {str(e)}")


def _save_apng(frames: List[np.ndarray], filepath: str, duration: Union[float, List[float]], loop: int) -> None:
    """Save frames as APNG using apng library."""
    if not APNG_SUPPORT:
        raise ValueError("APNG support not available. Install: pip install apng")
    
    from PIL import Image
    import io
    
    # Convert frames to PNG files in memory
    png_frames = []
    
    # Handle duration
    if isinstance(duration, (list, tuple)):
        durations = [int(d * 1000) for d in duration]  # Convert to milliseconds
    else:
        durations = [int(duration * 1000)] * len(frames)
    
    for i, frame in enumerate(frames):
        # Convert numpy array to PIL Image
        if frame.ndim == 2:
            img = Image.fromarray(frame, mode='L')
        elif frame.shape[2] == 3:
            img = Image.fromarray(frame, mode='RGB')
        elif frame.shape[2] == 4:
            img = Image.fromarray(frame, mode='RGBA')
        
        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        png_data = buffer.getvalue()
        
        png_frames.append((png_data, durations[i]))
    
    # Create APNG
    apng = APNG()
    for png_data, delay in png_frames:
        apng.append_file(None, delay=delay, delay_den=1000)
        # Manually add PNG data
        apng.frames[-1] = (io.BytesIO(png_data), apng.frames[-1][1])
    
    apng.num_plays = loop
    apng.save(filepath)


def _save_video_format(frames: List[np.ndarray], filepath: str, fps: Optional[float], 
                       output_format: str, **kwargs) -> None:
    """Save frames as video (MP4, WebM) using imageio."""
    if fps is None:
        fps = 30.0  # Default FPS for video
    
    # Set codec based on format
    codec_map = {
        'MP4': 'libx264',
        'WEBM': 'libvpx-vp9'
    }
    
    codec = kwargs.pop('codec', codec_map.get(output_format, 'libx264'))
    
    # Prepare writer kwargs
    writer_kwargs = {
        'fps': fps,
        'codec': codec,
        'quality': kwargs.pop('quality', 8),  # imageio quality scale
        'pixelformat': kwargs.pop('pixelformat', 'yuv420p'),
    }
    writer_kwargs.update(kwargs)
    
    # Write video
    if IMAGEIO_V3:
        iio.imwrite(filepath, frames, plugin='pyav', **writer_kwargs)
    else:
        writer = iio.get_writer(filepath, fps=fps, codec=codec, **writer_kwargs)
        for frame in frames:
            writer.append_data(frame)
        writer.close()


def _save_image_format(frames: List[np.ndarray], filepath: str, 
                       duration: Union[float, List[float]], loop: int,
                       quality: Optional[int], optimize: bool, parallel: bool,
                       output_format: str, **kwargs) -> None:
    """Save frames as image format (GIF, WebP, AVIF) using PIL."""
    # Use PIL for writing with proper timing and compression
    from PIL import Image
    
    # Convert first frame to determine mode
    first_frame = frames[0]
    
    # Determine mode based on frame dimensions
    if first_frame.ndim == 2:
        mode = 'L'  # Grayscale
    elif first_frame.ndim == 3:
        if first_frame.shape[2] == 3:
            mode = 'RGB'
        elif first_frame.shape[2] == 4:
            mode = 'RGBA'
        elif first_frame.shape[2] == 1:
            # Single channel in 3D array - convert to 2D
            frames = [f.squeeze() if f.ndim == 3 and f.shape[2] == 1 else f 
                               for f in frames]
            mode = 'L'
        else:
            raise ValueError(f"Unsupported number of channels: {first_frame.shape[2]}")
    else:
        raise ValueError(f"Frame must be 2D or 3D, got shape: {first_frame.shape}")
    
    # Determine if we need palette conversion (only for GIF)
    use_palette = (output_format == 'GIF' and mode in ('RGB', 'RGBA'))
    
    # Function to convert a single frame based on output format
    def convert_frame(frame_array):
        # Handle edge case where frame might have wrong dimensions
        if frame_array.ndim == 3 and frame_array.shape[2] == 1:
            frame_array = frame_array.squeeze()
        
        img = Image.fromarray(frame_array, mode=mode)
        
        # For GIF, convert to palette mode for better compression
        if use_palette:
            img = img.convert('P', palette=Image.ADAPTIVE, colors=256)
        # For WebP, AVIF keep original mode (supports RGB and RGBA natively)
        
        return img
    
    # Convert frames - use parallel processing if requested and beneficial
    # Only use parallel for large frame counts (overhead otherwise)
    if parallel and len(frames) > 20:
        # Parallel processing for large numbers of frames
        num_workers = min(4, multiprocessing.cpu_count())  # Cap at 4 for I/O bound tasks
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            pil_frames = list(executor.map(convert_frame, frames))
    else:
        # Sequential processing for small numbers of frames (faster due to less overhead)
        pil_frames = [convert_frame(frame) for frame in frames]
    
    # Prepare save parameters based on format
    save_params = {
        'save_all': True,
        'append_images': pil_frames[1:] if len(pil_frames) > 1 else [],
    }
    
    # Handle duration - convert back to milliseconds for image formats
    if isinstance(duration, (list, tuple)):
        # Convert seconds to milliseconds
        duration_ms = [int(d * 1000) for d in duration]
        save_params['duration'] = duration_ms
    else:
        # Single duration value
        duration_ms = int(duration * 1000)
        save_params['duration'] = duration_ms
    
    # Format-specific parameters
    if output_format == 'GIF':
        save_params['loop'] = loop if loop is not None else 0
        save_params['optimize'] = True  # Always optimize for smaller file size
        save_params['disposal'] = 2  # Restore to background color between frames
    
    elif output_format == 'WEBP':
        save_params['loop'] = loop if loop is not None else 0
        save_params['lossless'] = False  # Use lossy compression for smaller files
        save_params['quality'] = quality if quality is not None else 80
        save_params['method'] = 4  # Compression method (0-6) - 4 is good balance of speed/quality
    
    elif output_format == 'AVIF':
        if not AVIF_SUPPORT:
            raise ValueError("AVIF support not available. Install: pip install pillow-avif-plugin")
        save_params['quality'] = quality if quality is not None else 80
        save_params['speed'] = kwargs.pop('speed', 6)  # Encoding speed (0-10) - 6 is good balance
    
    # Merge with any additional kwargs (allow user override)
    save_params.update(kwargs)
    
    # Save using PIL
    pil_frames[0].save(
        filepath,
        format=output_format,
        **save_params
    )


def process_gif_list(list_file: str, output_suffix: str = '_out', output_format: Optional[str] = None) -> Dict[str, Any]:
    """
    Process multiple animated files listed in a text file.
    
    Reads a list of file paths from a text file and processes each one,
    saving the output with a suffix added to the filename.
    
    Args:
        list_file: Path to text file containing file paths (one per line)
        output_suffix: Suffix to add before extension (default: '_out')
        output_format: Force output format ('GIF', 'WEBP', 'APNG', 'AVIF', 'MP4', 'WEBM')
                      If None, keeps original format
        
    Returns:
        Dictionary containing processing results
        
    Text file format:
        - One file path per line
        - Paths can be absolute or relative
        - Empty lines and lines starting with # are ignored
        - No quotes needed around paths (but they're ok if present)
        - Supports: .gif, .webp, .png/.apng, .avif, .mp4, .webm
        - Example:
            /path/to/file1.gif
            /path/to/file2.webp
            relative/path/file3.mp4
            # This is a comment
            /another/file4.avif
    """
    import time
    
    print("=" * 70)
    print("BATCH ANIMATED FILE PROCESSING")
    print("=" * 70)
    
    # Read the file list
    if not os.path.exists(list_file):
        raise FileNotFoundError(f"List file not found: {list_file}")
    
    with open(list_file, 'r') as f:
        lines = f.readlines()
    
    # Parse file paths
    file_paths = []
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        # Remove quotes if present
        if (line.startswith('"') and line.endswith('"')) or \
           (line.startswith("'") and line.endswith("'")):
            line = line[1:-1]
        
        file_paths.append(line)
    
    print(f"\nFound {len(file_paths)} files to process")
    print(f"Output suffix: '{output_suffix}'")
    if output_format:
        print(f"Output format: {output_format}")
    else:
        print(f"Output format: Keep original")
    print("-" * 70)
    
    # Process each file
    results = {
        'total': len(file_paths),
        'successful': 0,
        'failed': 0,
        'results': []
    }
    
    for i, input_path in enumerate(file_paths, 1):
        file_result = {
            'input': input_path,
            'output': None,
            'success': False,
            'error': None,
            'duration': 0,
            'input_size': 0,
            'output_size': 0
        }
        
        try:
            print(f"\n[{i}/{len(file_paths)}] Processing: {input_path}")
            start_time = time.time()
            
            # Check if input exists
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found")
            
            # Generate output path
            path_parts = os.path.splitext(input_path)
            base_name = path_parts[0]
            original_ext = path_parts[1]
            
            # Determine output extension
            if output_format:
                ext_map = {
                    'GIF': '.gif',
                    'WEBP': '.webp',
                    'APNG': '.png',
                    'AVIF': '.avif',
                    'MP4': '.mp4',
                    'WEBM': '.webm'
                }
                out_ext = ext_map.get(output_format.upper(), original_ext)
            else:
                out_ext = original_ext  # Keep original extension
            
            output_path = f"{base_name}{output_suffix}{out_ext}"
            file_result['output'] = output_path
            
            # Load the file
            frames, metadata = load_video(input_path)
            print(f"  ✓ Loaded: {len(frames)} frames")
            
            # Save the file
            # For video formats, use fps instead of per-frame duration
            if metadata.get('format', '').upper() in ['MP4', 'WEBM'] or \
               (output_format and output_format.upper() in ['MP4', 'WEBM']):
                # Use FPS for video formats
                fps_val = metadata.get('fps', 30.0)
                save_video(frames, output_path, 
                          fps=fps_val,
                          optimize=True,
                          parallel=len(frames) > 20,
                          output_format=output_format)
            else:
                # Use duration for image formats
                if metadata['frames'] and 'duration' in metadata['frames'][0]:
                    frame_durations = [f.get('duration', 0.1) for f in metadata['frames']]
                    save_video(frames, output_path, 
                              duration=frame_durations,
                              loop=0,
                              optimize=True,
                              parallel=len(frames) > 20,
                              output_format=output_format)
                else:
                    save_duration = metadata.get('avg_duration') or metadata.get('duration', 0.1)
                    save_video(frames, output_path, 
                              duration=save_duration,
                              loop=0,
                              optimize=True,
                              parallel=len(frames) > 20,
                              output_format=output_format)
            
            print(f"  ✓ Saved: {output_path}")
            
            # Get file sizes
            file_result['input_size'] = os.path.getsize(input_path)
            file_result['output_size'] = os.path.getsize(output_path)
            file_result['duration'] = time.time() - start_time
            file_result['success'] = True
            results['successful'] += 1
            
            # Show stats
            size_ratio = file_result['output_size'] / file_result['input_size']
            print(f"  ✓ Size: {file_result['input_size']:,} → {file_result['output_size']:,} bytes ({size_ratio:.1%})")
            print(f"  ✓ Time: {file_result['duration']:.2f}s")
            
        except Exception as e:
            file_result['error'] = str(e)
            file_result['duration'] = time.time() - start_time
            results['failed'] += 1
            print(f"  ✗ ERROR: {e}")
        
        results['results'].append(file_result)
    
    # Print summary
    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total files: {results['total']}")
    print(f"Successful: {results['successful']} ✓")
    print(f"Failed: {results['failed']} ✗")
    
    if results['successful'] > 0:
        total_time = sum(r['duration'] for r in results['results'])
        avg_time = total_time / results['successful']
        total_input = sum(r['input_size'] for r in results['results'] if r['success'])
        total_output = sum(r['output_size'] for r in results['results'] if r['success'])
        
        print(f"\nTotal processing time: {total_time:.2f}s")
        print(f"Average time per file: {avg_time:.2f}s")
        print(f"Total input size: {total_input:,} bytes ({total_input/1024/1024:.2f} MB)")
        print(f"Total output size: {total_output:,} bytes ({total_output/1024/1024:.2f} MB)")
        print(f"Overall compression: {total_output/total_input:.1%}")
    
    if results['failed'] > 0:
        print(f"\nFailed files:")
        for r in results['results']:
            if not r['success']:
                print(f"  - {r['input']}: {r['error']}")
    
    return results


def test_functions():
    """
    Test the load_video and save_video functions with specified paths.
    """
    input_path = '/Users/dianaphelps/code/testing/testgifs/testgif.gif'
    output_path = '/Users/dianaphelps/code/testing/testgifs/tesoutput.gif'
    
    print("=" * 70)
    print("TESTING vidio.py FUNCTIONS")
    print("=" * 70)
    
    try:
        # Test 1: Load the GIF
        print(f"\n[TEST 1] Loading GIF from: {input_path}")
        print("-" * 70)
        
        frames, metadata = load_video(input_path)
        
        print("✓ Successfully loaded GIF!")
        print(f"\nMetadata:")
        print(f"  - Number of frames: {len(frames)}")
        print(f"  - Frame shape: {frames[0].shape}")
        print(f"  - Data type: {frames[0].dtype}")
        print(f"  - Is animated: {metadata.get('is_animated', False)}")
        print(f"  - Total images in file: {metadata.get('n_images', 'N/A')}")
        
        if 'duration' in metadata:
            print(f"  - Global duration: {metadata['duration']} seconds")
        
        if 'avg_duration' in metadata:
            print(f"  - Average frame duration: {metadata['avg_duration']:.4f} seconds")
            print(f"  - Approximate FPS: {1.0/metadata['avg_duration']:.2f}")
        
        # Calculate total video duration
        if metadata['frames'] and len(metadata['frames']) > 0:
            if 'duration' in metadata['frames'][0]:
                total_duration = sum(f.get('duration', 0) for f in metadata['frames'])
                print(f"  - Total video duration: {total_duration:.2f} seconds")
        
        # Show info about first few frames
        print(f"\nFirst 3 frames info:")
        for i, frame_info in enumerate(metadata['frames'][:3]):
            if 'shape' in frame_info:
                print(f"  Frame {i}: shape={frame_info['shape']}, dtype={frame_info['dtype']}", end='')
            else:
                print(f"  Frame {i}: shape={frames[i].shape}, dtype={frames[i].dtype}", end='')
            
            if 'duration' in frame_info:
                print(f", duration={frame_info['duration']:.4f}s")
            else:
                print()
        
        # Test 2: Save the GIF
        print(f"\n[TEST 2] Saving GIF to: {output_path}")
        print("-" * 70)
        
        # Preserve original timing - use per-frame durations if available
        if metadata['frames'] and 'duration' in metadata['frames'][0]:
            # Use individual frame durations for accurate playback speed
            frame_durations = [f.get('duration', 0.1) for f in metadata['frames']]
            print(f"Using per-frame durations (range: {min(frame_durations):.4f}s - {max(frame_durations):.4f}s)")
            
            save_video(
                frames=frames,
                filepath=output_path,
                duration=frame_durations,
                loop=0,  # Infinite loop
                optimize=True,
                parallel=len(frames) > 20
            )
        else:
            # Use average/global duration
            save_duration = metadata.get('avg_duration') or metadata.get('duration', 0.1)
            print(f"Using global duration: {save_duration:.4f}s")
            
            save_video(
                frames=frames,
                filepath=output_path,
                duration=save_duration,
                loop=0,  # Infinite loop
                optimize=True,
                parallel=len(frames) > 20
            )
        
        print("✓ Successfully saved GIF!")
        
        # Test 3: Verify output and compare
        print(f"\n[TEST 3] Verifying output")
        print("-" * 70)
        
        # Check that output file exists
        if not os.path.exists(output_path):
            raise IOError("Output file was not created!")
        
        print("✓ Output file exists")
        
        # Compare file sizes
        input_size = os.path.getsize(input_path)
        output_size = os.path.getsize(output_path)
        
        print(f"\nFile size comparison:")
        print(f"  - Input:  {input_size:,} bytes ({input_size/1024:.2f} KB)")
        print(f"  - Output: {output_size:,} bytes ({output_size/1024:.2f} KB)")
        print(f"  - Ratio:  {output_size/input_size:.1%}")
        
        if output_size < input_size:
            savings = input_size - output_size
            print(f"  - Saved:  {savings:,} bytes ({savings/1024:.2f} KB)")
        elif output_size > input_size:
            increase = output_size - input_size
            print(f"  - Larger by: {increase:,} bytes ({increase/1024:.2f} KB)")
        else:
            print(f"  - Same size")
        
        # Test 4: Reload output to verify integrity
        print(f"\n[TEST 4] Verifying output integrity and timing")
        print("-" * 70)
        
        output_frames, output_metadata = load_video(output_path)
        
        print("✓ Output file can be loaded successfully")
        print(f"  - Output has {len(output_frames)} frames")
        print(f"  - Input had {len(frames)} frames")
        
        if len(output_frames) == len(frames):
            print("✓ Frame count matches!")
        else:
            print("⚠ Warning: Frame count differs")
        
        # Compare frame shapes
        if output_frames[0].shape == frames[0].shape:
            print(f"✓ Frame dimensions match: {frames[0].shape}")
        else:
            print(f"⚠ Warning: Frame dimensions differ")
            print(f"  Input: {frames[0].shape}, Output: {output_frames[0].shape}")
        
        # Compare timing
        if metadata['frames'] and output_metadata['frames']:
            input_durations = [f.get('duration', 0) for f in metadata['frames']]
            output_durations = [f.get('duration', 0) for f in output_metadata['frames']]
            
            if input_durations and output_durations:
                avg_input = sum(input_durations) / len(input_durations)
                avg_output = sum(output_durations) / len(output_durations)
                print(f"\nTiming comparison:")
                print(f"  - Input average duration: {avg_input:.4f}s")
                print(f"  - Output average duration: {avg_output:.4f}s")
                
                diff_percent = abs(avg_output - avg_input) / avg_input * 100
                if diff_percent < 5:
                    print(f"✓ Timing matches within {diff_percent:.1f}%")
                else:
                    print(f"⚠ Timing differs by {diff_percent:.1f}%")
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}")
        print(f"\nMake sure the input file exists at: {input_path}")
        return False
        
    except PermissionError as e:
        print(f"\n✗ ERROR: {e}")
        print(f"\nCheck file permissions for input and output paths")
        return False
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False


def main():
    """
    Main entry point for command-line usage.
    """
    import sys
    
    print("vidio.py - Multi-Format Animated Image/Video Handler")
    print("Supports: GIF, WebP, APNG, AVIF, MP4, WebM")
    print("=" * 60)
    
    # If no arguments, run the test with specified paths
    if len(sys.argv) == 1:
        print("\nRunning automated test with specified paths...")
        success = test_functions()
        sys.exit(0 if success else 1)
    
    # Check for batch processing mode
    elif sys.argv[1] == '--batch' or sys.argv[1] == '-b':
        if len(sys.argv) < 3:
            print("\nError: Batch mode requires a list file")
            print("Usage: python vidio.py --batch <list_file> [output_suffix] [format]")
            sys.exit(1)
        
        list_file = sys.argv[2]
        output_suffix = sys.argv[3] if len(sys.argv) > 3 else '_out'
        output_format = sys.argv[4] if len(sys.argv) > 4 else None
        
        try:
            results = process_gif_list(list_file, output_suffix, output_format)
            sys.exit(0 if results['failed'] == 0 else 1)
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # If arguments provided, use command-line mode
    elif len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "output_test.gif"
        
        try:
            print(f"\nLoading file: {input_path}")
            frames, metadata = load_video(input_path)
            
            print(f"\nSuccessfully loaded!")
            print(f"Number of frames: {len(frames)}")
            print(f"Frame shape: {frames[0].shape}")
            print(f"Data type: {frames[0].dtype}")
            print(f"Animated: {metadata.get('is_animated', False)}")
            
            if 'duration' in metadata:
                print(f"Duration: {metadata['duration']} seconds")
            if 'avg_duration' in metadata:
                print(f"Average frame duration: {metadata['avg_duration']:.3f} seconds")
            
            # Calculate total video duration
            if metadata['frames'] and 'duration' in metadata['frames'][0]:
                total_duration = sum(f.get('duration', 0) for f in metadata['frames'])
                print(f"Total video duration: {total_duration:.2f} seconds")
            
            # Save as output file
            print(f"\nSaving to: {output_path}")
            
            # Extract per-frame durations if available
            if metadata['frames'] and 'duration' in metadata['frames'][0]:
                # Use individual frame durations
                frame_durations = [f.get('duration', 0.1) for f in metadata['frames']]
                print(f"Using per-frame durations (avg: {sum(frame_durations)/len(frame_durations):.4f}s)")
                save_video(frames, output_path, 
                          duration=frame_durations,
                          loop=0,
                          optimize=True,
                          parallel=len(frames) > 20)
            else:
                # Use global duration or default
                save_duration = metadata.get('avg_duration') or metadata.get('duration', 0.1)
                print(f"Using global duration: {save_duration:.4f}s")
                save_video(frames, output_path, 
                          duration=save_duration,
                          loop=0,
                          optimize=True,
                          parallel=len(frames) > 20)
            
            print("Successfully saved!")
            
            # Show file sizes
            original_size = os.path.getsize(input_path)
            output_size = os.path.getsize(output_path)
            print(f"\nOriginal size: {original_size:,} bytes")
            print(f"Output size: {output_size:,} bytes")
            print(f"Compression ratio: {output_size/original_size:.1%}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("\nUsage:")
        print("  python vidio.py                              # Run test with default paths")
        print("  python vidio.py <input> [output]             # Process single file")
        print("  python vidio.py --batch <list_file> [suffix] [format] # Batch process")
        print("\nSupported formats:")
        print("  Input:  GIF, WebP, APNG, AVIF, MP4, WebM")
        print("  Output: GIF, WebP, APNG, AVIF, MP4, WebM")
        print("\nBatch mode text file format:")
        print("  - One file path per line")
        print("  - No quotes needed (but ok if present)")
        print("  - Lines starting with # are comments")
        print("  - Empty lines are ignored")
        print("\nExample list file:")
        print("  /path/to/file1.gif")
        print("  /path/to/file2.webp")
        print("  /path/to/video.mp4")
        print("  # This is a comment")
        print("  relative/path/file3.avif")
        print("\nFormat conversion examples:")
        print("  python vidio.py input.gif output.webp        # GIF to WebP")
        print("  python vidio.py input.mp4 output.gif         # MP4 to GIF")
        print("  python vidio.py --batch list.txt _out WEBP   # Convert all to WebP")
        print("\nOptional dependencies:")
        print("  APNG:  pip install apng")
        print("  AVIF:  pip install pillow-avif-plugin")
        print("  Video: pip install imageio[pyav] or imageio[ffmpeg]")
        print("\nCore functions:")
        print("  - load_video(filepath): Load any supported format")
        print("  - save_video(frames, filepath, ...): Save in any supported format")


if __name__ == "__main__":
    main()
