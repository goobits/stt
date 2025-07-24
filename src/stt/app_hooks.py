"""
App hooks for STT CLI - bridges Goobits CLI with existing STT functionality
"""
import sys
from typing import List, Optional, Any

# Import existing STT modules
import sys
import os
# Add parent directory to path to import main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main as legacy_main


# Helper function to build command line args for legacy main
def _build_args(mode: str, **kwargs) -> List[str]:
    """Build command line arguments for legacy main function"""
    args = [mode]
    
    # Add options
    if 'model' in kwargs and kwargs['model']:
        args.extend(['--model', kwargs['model']])
    if 'language' in kwargs and kwargs['language']:
        args.extend(['--language', kwargs['language']])
    if 'device' in kwargs and kwargs['device']:
        args.extend(['--device', kwargs['device']])
    if 'no_formatting' in kwargs and kwargs['no_formatting']:
        args.append('--no-formatting')
    if 'json' in kwargs and kwargs['json']:
        args.append('--json')
    if 'debug' in kwargs and kwargs['debug']:
        args.append('--debug')
    if 'config' in kwargs and kwargs['config']:
        args.extend(['--config', kwargs['config']])
    if 'sample_rate' in kwargs and kwargs['sample_rate']:
        args.extend(['--sample-rate', str(kwargs['sample_rate'])])
    
    return args


# Recording mode commands
def on_listen(device: Optional[str] = None, language: Optional[str] = None, 
              model: Optional[str] = None, hold_to_talk: Optional[str] = None,
              json: bool = False, debug: bool = False, config: Optional[str] = None,
              no_formatting: bool = False, sample_rate: Optional[int] = None, **kwargs):
    """Listen once and transcribe"""
    # Build kwargs dict for _build_args
    options = {
        'device': device, 'language': language, 'model': model,
        'json': json, 'debug': debug, 'config': config,
        'no_formatting': no_formatting, 'sample_rate': sample_rate
    }
    args = _build_args('listen', **options)
    
    if hold_to_talk:
        args.extend(['--hold-to-talk', hold_to_talk])
    
    # Call legacy main with constructed args
    sys.argv = ['stt'] + args
    return legacy_main()


def on_live(device: Optional[str] = None, language: Optional[str] = None, 
            model: Optional[str] = None, tap_to_talk: Optional[str] = None,
            json: bool = False, debug: bool = False, config: Optional[str] = None,
            no_formatting: bool = False, sample_rate: Optional[int] = None, **kwargs):
    """Live conversation mode"""
    # Build kwargs dict for _build_args
    options = {
        'device': device, 'language': language, 'model': model,
        'json': json, 'debug': debug, 'config': config,
        'no_formatting': no_formatting, 'sample_rate': sample_rate
    }
    args = _build_args('live', **options)
    
    if tap_to_talk:
        args.extend(['--tap-to-talk', tap_to_talk])
    
    sys.argv = ['stt'] + args
    return legacy_main()




# Server and processing commands
def on_serve(port: int = 8769, host: str = "0.0.0.0", 
             debug: bool = False, config: Optional[str] = None, **kwargs):
    """Start transcription server"""
    args = ['serve']
    
    args.extend(['--port', str(port)])
    args.extend(['--host', host])
    
    if debug:
        args.append('--debug')
    if config:
        args.extend(['--config', config])
    
    sys.argv = ['stt'] + args
    return legacy_main()




# System commands
def on_status(**kwargs):
    """Show system status"""
    args = ['status']
    sys.argv = ['stt'] + args
    return legacy_main()


def on_models(**kwargs):
    """List available models"""
    args = ['models']
    sys.argv = ['stt'] + args
    return legacy_main()


# Configuration commands
def on_config_list(**kwargs):
    """List all configuration"""
    args = ['config', 'list']
    sys.argv = ['stt'] + args
    return legacy_main()


def on_config_get(key: str, **kwargs):
    """Get configuration value"""
    args = ['config', 'get', key]
    sys.argv = ['stt'] + args
    return legacy_main()


def on_config_set(key: str, value: str, **kwargs):
    """Set configuration value"""
    args = ['config', 'set', key, value]
    sys.argv = ['stt'] + args
    return legacy_main()




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