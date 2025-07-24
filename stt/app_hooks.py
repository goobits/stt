"""
App hooks for STT CLI - bridges Goobits CLI with existing STT functionality
"""
import asyncio
import sys
from typing import Optional, Any
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))


def create_args_object(**kwargs):
    """Create an args object that mimics argparse namespace"""
    class Args:
        pass
    
    args = Args()
    
    # Set all kwargs as attributes
    for key, value in kwargs.items():
        # Convert underscores to hyphens for consistency
        attr_name = key.replace('-', '_')
        setattr(args, attr_name, value)
    
    # Set defaults for common attributes
    if not hasattr(args, 'debug'):
        args.debug = False
    if not hasattr(args, 'json'):
        args.json = False
    if not hasattr(args, 'model'):
        args.model = 'base'
    if not hasattr(args, 'language'):
        args.language = None
    if not hasattr(args, 'device'):
        args.device = None
    if not hasattr(args, 'no_formatting'):
        args.no_formatting = False
    if not hasattr(args, 'sample_rate'):
        args.sample_rate = 16000
    if not hasattr(args, 'config'):
        args.config = None
    
    return args


# Recording mode commands
def on_listen(device: Optional[str] = None, language: Optional[str] = None, 
              model: Optional[str] = None, hold_to_talk: Optional[str] = None,
              json: bool = False, debug: bool = False, config: Optional[str] = None,
              no_formatting: bool = False, sample_rate: Optional[int] = None, **kwargs):
    """Listen once and transcribe"""
    from src.modes.listen_once import ListenOnceMode
    
    args = create_args_object(
        device=device, language=language, model=model,
        hold_to_talk=hold_to_talk, json=json, debug=debug,
        config=config, no_formatting=no_formatting,
        sample_rate=sample_rate or 16000
    )
    
    mode = ListenOnceMode(args)
    return asyncio.run(mode.run())


def on_live(device: Optional[str] = None, language: Optional[str] = None, 
            model: Optional[str] = None, tap_to_talk: Optional[str] = None,
            json: bool = False, debug: bool = False, config: Optional[str] = None,
            no_formatting: bool = False, sample_rate: Optional[int] = None, **kwargs):
    """Live conversation mode"""
    from src.modes.conversation import ConversationMode
    
    args = create_args_object(
        device=device, language=language, model=model,
        tap_to_talk=tap_to_talk, json=json, debug=debug,
        config=config, no_formatting=no_formatting,
        sample_rate=sample_rate or 16000
    )
    
    mode = ConversationMode(args)
    return asyncio.run(mode.run())


# Server command
def on_serve(port: int = 8769, host: str = "0.0.0.0", 
             debug: bool = False, config: Optional[str] = None, **kwargs):
    """Start transcription server"""
    from src.transcription.server import TranscriptionServer
    
    args = create_args_object(
        port=port, host=host, debug=debug, config=config
    )
    
    # Server mode is a bit different - it starts directly
    server = TranscriptionServer(port=port, host=host)
    try:
        asyncio.run(server.start())
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        if debug:
            import traceback
            traceback.print_exc()
        print(f"Server error: {e}")
        return 1


# System commands
def on_status(**kwargs):
    """Show system status"""
    try:
        from src.utils.system_status import show_system_status
        show_system_status()
        return 0
    except ImportError:
        print("System status module not available")
        return 1


def on_models(**kwargs):
    """List available models"""
    print("Available Whisper models:")
    print("  tiny    - Fastest, least accurate (39M parameters)")
    print("  base    - Fast, good accuracy (74M parameters)")
    print("  small   - Balanced speed/accuracy (244M parameters)")
    print("  medium  - Slower, better accuracy (769M parameters)")
    print("  large   - Slowest, best accuracy (1550M parameters)")
    return 0


# Configuration commands
def on_config_list(**kwargs):
    """List all configuration"""
    from src.core.config import Config
    
    config = Config()
    config.list_all()
    return 0


def on_config_get(key: str, **kwargs):
    """Get configuration value"""
    from src.core.config import Config
    
    config = Config()
    value = config.get(key)
    if value is not None:
        print(f"{key}: {value}")
        return 0
    else:
        print(f"Configuration key '{key}' not found")
        return 1


def on_config_set(key: str, value: str, **kwargs):
    """Set configuration value"""
    from src.core.config import Config
    
    config = Config()
    try:
        config.set(key, value)
        print(f"Set {key} = {value}")
        return 0
    except Exception as e:
        print(f"Error setting configuration: {e}")
        return 1


# Hook for custom initialization if needed
def initialize_app():
    """Initialize the STT application"""
    # Any initialization code needed before commands run
    pass


# Hook for custom cleanup if needed  
def cleanup_app():
    """Cleanup when app exits"""
    # Any cleanup code needed
    pass